from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


def _inventory_tool() -> ModuleType:
    tool_path = REPO_ROOT / "tools" / "inventory_public_api.py"
    spec = importlib.util.spec_from_file_location(
        "inventory_public_api_contract_test",
        tool_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_named(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in {path}")


def test_discovered_submodule_is_parsed_without_import_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    inventory = _inventory_tool()
    package_root = tmp_path / "fast_mlsirm"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    sentinel = tmp_path / "import-side-effect.txt"
    (package_root / "hostile.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n"
        "def public_contract(value=1):\n"
        "    return value\n",
        encoding="utf-8",
    )

    inventory._install_core_stub()
    import fast_mlsirm

    monkeypatch.setattr(fast_mlsirm, "__path__", [str(package_root)])
    monkeypatch.setattr(inventory, "PY_ROOT", tmp_path)
    if hasattr(inventory, "PACKAGE_ROOT"):
        monkeypatch.setattr(inventory, "PACKAGE_ROOT", package_root)
    sys.modules.pop("fast_mlsirm.hostile", None)

    rows = inventory.collect_python_rows()

    assert not sentinel.exists()
    assert {
        "source": "python",
        "current_name": "public_contract",
        "module": "fast_mlsirm.hostile",
        "kind": "function",
        "parameters": "value=1",
    } in rows


def test_static_dataclass_projection_matches_rag_anchor_constructor_contract() -> None:
    inventory = _inventory_tool()
    anchor = _class_named(
        REPO_ROOT / "python" / "fast_mlsirm" / "scoring" / "rag.py",
        "RAGPerturbationAnchor",
    )

    assert inventory._ast_class_params(anchor) == (
        "anchor_id, baseline_request_fingerprint, perturbed_request_fingerprint, "
        "perturbation_specification_fingerprint, perturbation_run_fingerprint, "
        "perturbation_kind, _anchor_token=None"
    )


def test_static_dataclass_projection_respects_classvar_initvar_and_field_init() -> None:
    inventory = _inventory_tool()
    tree = ast.parse(
        '''
@dataclass(frozen=True)
class Contract:
    schema_version: ClassVar[str] = "v1"
    value: str
    derived: str = field(init=False)
    generated: str = field(default_factory=str)
    _token: InitVar[object | None] = None
'''
    )
    contract = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    assert inventory._ast_class_params(contract) == (
        "value, generated=<factory>, _token=None"
    )

def test_static_dataclass_projection_respects_disabled_generated_initializer() -> None:
    inventory = _inventory_tool()
    tree = ast.parse(
        """
@dataclass(init=False)
class Contract:
    value: str
"""
    )
    contract = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    assert inventory._ast_class_params(contract) == ""


def test_static_dataclass_field_without_default_remains_required() -> None:
    inventory = _inventory_tool()
    tree = ast.parse(
        """
@dataclass
class Contract:
    value: str = field(repr=False)
"""
    )
    contract = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    assert inventory._ast_class_params(contract) == "value"
