#!/usr/bin/env python3
"""Regenerate docs/api/inventory-<date>.csv.

Inventories every public callable exported by ``python/fast_mlsirm``
(top-level package surface, ``_legacy_init.py`` re-exports, and every
submodule) plus the PyO3 entry points compiled from
``crates/fast-mlsirm-py``. Repository module discovery is static: the tool
imports only the fixed ``fast_mlsirm`` package surface after installing a
stub ``fast_mlsirm._core`` and parses any otherwise-unloaded submodule from
source instead of importing a discovered module name. It does not require a
built Rust extension and does not invoke cargo/maturin.

Usage:
    python tools/inventory_public_api.py [--date YYYYMMDD]
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import re
import sys
import types
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_ROOT = REPO_ROOT / "python"
PACKAGE_ROOT = PY_ROOT / "fast_mlsirm"
RUST_SRC = REPO_ROOT / "crates" / "fast-mlsirm-py" / "src"
FIELDS = ["source", "current_name", "module", "kind", "parameters"]


def _install_core_stub() -> None:
    """Inject a fake ``fast_mlsirm._core`` so the package imports without cargo."""

    class _FakeCore(types.ModuleType):
        def __getattr__(self, name: str):  # noqa: ANN001 - dynamic stub
            def _stub(*_args, **_kwargs):
                raise RuntimeError("fast_mlsirm._core stub: Rust extension not built")

            return _stub

    sys.modules.setdefault("fast_mlsirm._core", _FakeCore("fast_mlsirm._core"))


def _format_default(value: object) -> str:
    if value is inspect.Parameter.empty:
        return ""
    try:
        return repr(value)
    except Exception:
        return "<unrepr-able>"


def _format_params(sig: inspect.Signature) -> str:
    parts = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{name}")
        else:
            default = _format_default(p.default)
            parts.append(f"{name}={default}" if default else name)
    return ", ".join(parts)


def _module_name(py_file: Path) -> str:
    """Return the import name represented by one source file under ``PY_ROOT``."""
    relative = py_file.relative_to(PY_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_public_module(module_name: str) -> bool:
    """Whether a discovered source module is public by package naming convention."""
    parts = module_name.split(".")[1:]
    return bool(parts) and all(not part.startswith("_") for part in parts)


def _ast_default(node: ast.expr | None) -> str:
    """Render a source default closely enough for the API inventory contract."""
    if node is None:
        return ""
    if isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name == "field":
            for keyword in node.keywords:
                if keyword.arg == "default_factory":
                    return "<factory>"
                if keyword.arg == "default":
                    return ast.unparse(keyword.value)
            return ""
    return ast.unparse(node)


def _ast_function_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Project a Python function declaration into the inventory parameter format."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(
        args.defaults
    )
    parts: list[str] = []
    for argument, default in zip(positional, defaults, strict=True):
        if argument.arg == "self":
            continue
        rendered_default = _ast_default(default)
        parts.append(
            f"{argument.arg}={rendered_default}" if rendered_default else argument.arg
        )
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    for argument, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        rendered_default = _ast_default(default)
        parts.append(
            f"{argument.arg}={rendered_default}" if rendered_default else argument.arg
        )
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "dataclass":
            return True
    return False


def _dataclass_generates_initializer(node: ast.ClassDef) -> bool:
    """Whether the dataclass decorator enables its generated initializer."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        is_dataclass_decorator = (
            isinstance(target, ast.Name) and target.id == "dataclass"
        ) or (
            isinstance(target, ast.Attribute) and target.attr == "dataclass"
        )
        if not is_dataclass_decorator:
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "init" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value is not False
    return True


def _annotation_root_name(annotation: ast.expr) -> str:
    """Return the outer annotation name used by dataclass constructor semantics."""
    while isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return ""


def _dataclass_field_participates_in_init(statement: ast.AnnAssign) -> bool:
    """Whether an annotated dataclass field participates in generated ``__init__``."""
    if _annotation_root_name(statement.annotation) == "ClassVar":
        return False
    value = statement.value
    if not isinstance(value, ast.Call):
        return True
    target = value.func
    if isinstance(target, ast.Name):
        call_name = target.id
    elif isinstance(target, ast.Attribute):
        call_name = target.attr
    else:
        call_name = ""
    if call_name != "field":
        return True
    for keyword in value.keywords:
        if keyword.arg == "init" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is not False
    return True


def _ast_class_params(node: ast.ClassDef) -> str:
    """Project explicit or dataclass-generated constructors without importing the module."""
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == "__init__":
            return _ast_function_params(statement)
    if not _is_dataclass(node) or not _dataclass_generates_initializer(node):
        return ""

    parts: list[str] = []
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        if not _dataclass_field_participates_in_init(statement):
            continue
        name = statement.target.id
        default = _ast_default(statement.value)
        parts.append(f"{name}={default}" if default else name)
    return ", ".join(parts)


def collect_python_rows() -> list[dict]:
    sys.path.insert(0, str(PY_ROOT))
    _install_core_stub()
    import fast_mlsirm  # noqa: F401  (fixed package import after path/stub setup)

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def visit(obj_name: str, obj: object) -> None:
        if obj_name.startswith("_"):
            return
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            return
        module = getattr(obj, "__module__", "") or ""
        if not module.startswith("fast_mlsirm"):
            return
        qualname = getattr(obj, "__qualname__", obj_name)
        key = (module, qualname)
        if key in seen:
            return
        seen.add(key)
        kind = "function" if inspect.isfunction(obj) else "class"
        try:
            params = _format_params(inspect.signature(obj))
        except (ValueError, TypeError):
            params = "<no-signature>"
        rows.append(
            {
                "source": "python",
                "current_name": obj_name,
                "module": module,
                "kind": kind,
                "parameters": params,
            }
        )

    def visit_static(module_name: str, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if node.name.startswith("_"):
            return
        key = (module_name, node.name)
        if key in seen:
            return
        seen.add(key)
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        params = _ast_class_params(node) if isinstance(node, ast.ClassDef) else _ast_function_params(node)
        rows.append(
            {
                "source": "python",
                "current_name": node.name,
                "module": module_name,
                "kind": kind,
                "parameters": params,
            }
        )

    # Top-level package surface includes explicit re-exports from private
    # implementation modules. This import target is fixed, never discovered
    # from caller-controlled data.
    for name in sorted(dir(fast_mlsirm)):
        visit(name, getattr(fast_mlsirm, name))

    # Enumerate repository-owned module paths from the filesystem. If a
    # module was already loaded by the fixed package import, inspect that
    # object. Otherwise parse its top-level declarations instead of
    # executing a discovered module name.
    for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
        module_name = _module_name(py_file)
        if not _is_public_module(module_name):
            continue
        loaded_module = sys.modules.get(module_name)
        if loaded_module is not None:
            for name in sorted(vars(loaded_module)):
                if name.startswith("_"):
                    continue
                obj = getattr(loaded_module, name)
                if getattr(obj, "__module__", None) != module_name:
                    continue
                visit(name, obj)
            continue

        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit_static(module_name, statement)

    rows.sort(key=lambda r: (r["module"], r["current_name"]))
    return rows


_PYMODULE_RE = re.compile(
    r'#\[pymodule\]\s*(?:#\[pyo3\(name\s*=\s*"([^"]+)"\)\]\s*)?'
    r"(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)\s*->\s*PyResult<\(\)>\s*\{"
)
_WRAP_PYFUNCTION_RE = re.compile(r"wrap_pyfunction!\((\w+)")
_SIGNATURE_ATTR_RE = re.compile(r"#\[pyo3\(signature\s*=\s*\((.*?)\)\)\]", re.DOTALL)
_FN_DEF_RE = re.compile(r"fn\s+{name}\s*(?:<[^>]*>)?\s*\((.*?)\)\s*(?:->|\{{)", re.DOTALL)


def _extract_braced_block(text: str, open_brace_index: int) -> str:
    depth = 0
    for i in range(open_brace_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index : i + 1]
    return text[open_brace_index:]


def _rust_params_from_signature_attr(params_blob: str) -> str:
    parts = [p.strip() for p in params_blob.strip().split(",") if p.strip()]
    return ", ".join(parts)


def _rust_params_from_fn_decl(params_blob: str) -> str:
    parts = []
    depth = 0
    current = ""
    for ch in params_blob:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    names = []
    for p in parts:
        p = p.strip()
        if not p or p in ("&self", "self"):
            continue
        name = p.split(":", 1)[0].strip()
        names.append(name)
    return ", ".join(names)


def collect_rust_rows() -> list[dict]:
    rows: list[dict] = []
    for rs_file in sorted(RUST_SRC.glob("*.rs")):
        text = rs_file.read_text()
        for match in _PYMODULE_RE.finditer(text):
            pymodule_name = match.group(1) or match.group(2)
            body = _extract_braced_block(text, match.end() - 1)
            fn_names = _WRAP_PYFUNCTION_RE.findall(body)
            for fn_name in fn_names:
                fn_def_match = re.search(
                    _FN_DEF_RE.pattern.format(name=re.escape(fn_name)), text, re.DOTALL
                )
                if fn_def_match is None:
                    params = "<not-found>"
                else:
                    preceding = text[: fn_def_match.start()]
                    sig_match = None
                    for sig_match in _SIGNATURE_ATTR_RE.finditer(preceding):
                        pass  # take the last (closest) match before fn decl
                    # Nothing but whitespace/other attrs sits between the closest
                    # #[pyo3(signature = (...))] and the "fn <name>(" it governs.
                    if sig_match is not None and preceding[sig_match.end() :].strip() in (
                        "",
                        "#[allow(clippy::too_many_arguments)]",
                    ):
                        params = _rust_params_from_signature_attr(sig_match.group(1))
                    else:
                        params = _rust_params_from_fn_decl(fn_def_match.group(1))
                rows.append(
                    {
                        "source": "pyo3",
                        "current_name": fn_name,
                        "module": f"fast_mlsirm.{pymodule_name}",
                        "kind": "function",
                        "parameters": params,
                    }
                )
    rows.sort(key=lambda r: (r["module"], r["current_name"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    rows = collect_python_rows() + collect_rust_rows()

    out_dir = REPO_ROOT / "docs" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inventory-{args.date}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
