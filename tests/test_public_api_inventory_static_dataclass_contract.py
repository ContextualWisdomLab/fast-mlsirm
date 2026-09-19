from __future__ import annotations

import ast
import importlib.util
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
