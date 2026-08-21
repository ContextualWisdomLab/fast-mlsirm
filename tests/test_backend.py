import pytest

from fast_mlsirm import backend


class _HostileControlName:
    """Object whose string conversion must never run at a control boundary."""

    def __str__(self) -> str:
        raise AssertionError("caller-controlled __str__ executed")


class _HostileControlString(str):
    """String subclass whose normalization callbacks must never run."""

    def __str__(self) -> str:
        raise AssertionError("caller-controlled str-subclass __str__ executed")

    def strip(self, chars=None) -> str:
        raise AssertionError("caller-controlled str-subclass strip executed")


def test_auto_unavailable_message_names_purchaser_next_action() -> None:
    """The fail-closed auto error must tell a purchaser what to install or pass."""
    message = backend.AUTO_BACKEND_UNAVAILABLE_MESSAGE

    assert message.startswith("compiled Rust core is required for automatic backend resolution")
    assert "fast_mlsirm._core" in message
    assert "backend='numpy'" in message
    assert ":" not in message
    assert "\\" not in message
    assert "/home/" not in message
    assert "/usr/" not in message
    assert "/tmp/" not in message


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

    with pytest.raises(RuntimeError, match="compiled Rust core is required") as exc_info:
        backend.resolve_backend("auto")

    assert str(exc_info.value) == backend.AUTO_BACKEND_UNAVAILABLE_MESSAGE
    assert "backend='numpy'" in str(exc_info.value)


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


def test_backend_normalizers_reject_untrusted_control_types_before_callbacks() -> None:
    """Backend/device controls must not invoke caller-defined string callbacks."""
    for normalizer, label in (
        (backend.normalize_backend, "backend"),
        (backend.normalize_device, "rust_device"),
    ):
        for value in (_HostileControlName(), _HostileControlString("auto")):
            with pytest.raises(ValueError, match=rf"{label} must be an exact built-in string"):
                normalizer(value)


def test_resolve_backend_rejects_untrusted_name_before_core_access(monkeypatch) -> None:
    """Public resolution must reject an untrusted control before native discovery."""
    core_loads: list[None] = []

    def unexpected_core_load():
        core_loads.append(None)
        raise AssertionError("native discovery must not run for an invalid backend control")

    monkeypatch.setattr(backend, "_load_core", unexpected_core_load)

    with pytest.raises(ValueError, match="backend must be an exact built-in string"):
        backend.resolve_backend(_HostileControlName())

    assert core_loads == []
