"""Fail-closed contracts for an installed but unloadable Rust extension."""

from __future__ import annotations

import importlib.machinery

import pytest

import fast_mlsirm.backend as backend


def _installed_core_that_fails_to_import(monkeypatch: pytest.MonkeyPatch) -> ImportError:
    """Make discovery succeed while the compiled extension import fails."""
    spec = importlib.machinery.ModuleSpec(backend.CORE_MODULE, loader=None)
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda _name: spec)
    failure = ImportError("dlopen failed: incompatible compiled extension")

    def fail_import(_name: str):
        raise failure

    monkeypatch.setattr(backend.importlib, "import_module", fail_import)
    return failure


@pytest.mark.parametrize("requested", ["auto", "rust"])
def test_resolve_backend_normalizes_compiled_core_import_failure(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    """Backend resolution must fail closed with a package-owned RuntimeError."""
    failure = _installed_core_that_fails_to_import(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="compiled Rust core is present but could not be imported",
    ) as caught:
        backend.resolve_backend(requested)

    assert caught.value.__cause__ is failure


def test_load_rust_core_normalizes_compiled_core_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct core loading must expose the same stable fail-closed contract."""
    failure = _installed_core_that_fails_to_import(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="compiled Rust core is present but could not be imported",
    ) as caught:
        backend.load_rust_core()

    assert caught.value.__cause__ is failure
