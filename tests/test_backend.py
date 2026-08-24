import pytest

from fast_mlsirm import backend


def test_load_core_surfaces_import_errors(monkeypatch):
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: object())

    def fail_import(name):
        raise RuntimeError("extension ABI mismatch")

    monkeypatch.setattr(backend.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="extension ABI mismatch"):
        backend._load_core()


def test_auto_backend_fails_closed_when_rust_core_is_unavailable(monkeypatch):
    """Automatic production resolution must not silently select NumPy."""
    monkeypatch.setattr(backend, "_load_core", lambda: None)

    with pytest.raises(RuntimeError, match="compiled Rust core is required"):
        backend.resolve_backend("auto")


def test_auto_backend_resolves_to_rust_when_core_is_available(monkeypatch):
    """Automatic resolution uses Rust when the compiled core is available."""
    sentinel_core = object()
    monkeypatch.setattr(backend, "_load_core", lambda: sentinel_core)

    assert backend.resolve_backend("auto") == "rust"


def test_explicit_numpy_reference_backend_remains_explicit(monkeypatch):
    """The explicit NumPy reference path must never depend on core discovery."""
    calls: list[None] = []

    def unexpected_core_load():
        calls.append(None)
        raise AssertionError("explicit NumPy resolution must not inspect the Rust core")

    monkeypatch.setattr(backend, "_load_core", unexpected_core_load)

    assert backend.resolve_backend("numpy") == "numpy"
    assert calls == []
