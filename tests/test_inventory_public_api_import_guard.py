"""The API inventory may import only public modules that live in this package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "inventory_public_api.py"


def _inventory():
    spec = importlib.util.spec_from_file_location("inventory_public_api_under_test", _TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_refuses_imports_outside_the_package():
    inventory = _inventory()
    refused = (
        "os",
        "fast_mlsirm",
        "fast_mlsirm._core",
        "fast_mlsirm.NotValid",
        "fast_mlsirm.missing_module_name",
        "fast_mlsirm/../os",
    )
    for name in refused:
        with pytest.raises((ValueError, ImportError)):
            inventory._import_owned_fast_mlsirm_module(name)


def test_inventory_imports_only_after_the_owned_source_check(monkeypatch):
    inventory = _inventory()
    seen: dict[str, str] = {}

    def fake_import(name: str):
        seen["name"] = name
        return object()

    monkeypatch.setattr(inventory.importlib, "import_module", fake_import)
    loaded = inventory._import_owned_fast_mlsirm_module("fast_mlsirm.dif")
    assert seen["name"] == "fast_mlsirm.dif"
    assert loaded is not None
