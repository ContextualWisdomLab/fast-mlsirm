"""Public-API inventory stays inside the repository tree."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _inventory():
    spec = importlib.util.spec_from_file_location(
        "inventory_public_api_under_test",
        ROOT / "tools" / "inventory_public_api.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("inventory loader spec was not created")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_refuses_names_outside_the_package():
    inventory = _inventory()
    for name in (
        "os",
        "fast_mlsirm..os",
        "fast_mlsirm.not-a-module",
        "fast_mlsirm.missing_module_name",
    ):
        with pytest.raises(ImportError):
            inventory._load_package_module(name)


def test_semgrep_targets_do_not_use_dynamic_globals_or_import_module():
    inventory = (ROOT / "tools" / "inventory_public_api.py").read_text(encoding="utf-8")
    dif = (ROOT / "python" / "fast_mlsirm" / "dif.py").read_text(encoding="utf-8")
    assert "import_module(" not in inventory
    assert "globals()" not in inventory
    assert "globals()" not in dif
