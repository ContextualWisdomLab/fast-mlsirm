import pytest

from fast_mlsirm import backend


class _HostileText(str):
    """String subclass that records any package-triggered text callback."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.callbacks = []
        return instance

    def __str__(self):
        self.callbacks.append("__str__")
        raise AssertionError("backend admission must not call __str__")

    def strip(self, *args, **kwargs):
        self.callbacks.append("strip")
        raise AssertionError("backend admission must not call strip")

    def lower(self):
        self.callbacks.append("lower")
        raise AssertionError("backend admission must not call lower")


class _HostileStringProvider:
    """Arbitrary object whose string conversion must never be used for admission."""

    def __init__(self):
        self.callbacks = []

    def __str__(self):
        self.callbacks.append("__str__")
        raise AssertionError("backend admission must not coerce arbitrary objects")


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


def test_explicit_numpy_reference_backend_requires_named_reference_scope(monkeypatch):
    """NumPy resolution is available only inside the named reference API scope."""
    calls: list[None] = []

    def unexpected_core_load():
        calls.append(None)
        raise AssertionError("explicit NumPy resolution must not inspect the Rust core")

    monkeypatch.setattr(backend, "_load_core", unexpected_core_load)

    with pytest.raises(RuntimeError, match="fit_reference"):
        backend.resolve_reference_backend()

    with backend._reference_backend_scope():
        assert backend.resolve_reference_backend() == "numpy"

    with pytest.raises(RuntimeError, match="fit_reference"):
        backend.resolve_reference_backend()
    assert calls == []


def test_production_backend_rejects_numpy_reference_name():
    """Production resolution cannot select the Python numerical owner."""
    with pytest.raises(ValueError, match="production backend"):
        backend.resolve_backend("numpy")


@pytest.mark.parametrize(
    ("normalizer", "raw"),
    [
        (backend.normalize_backend, "auto"),
        (backend.normalize_production_backend, "rust"),
        (backend.normalize_device, "cpu"),
        (backend.resolve_reference_backend, "numpy"),
    ],
)
def test_backend_semantic_controls_reject_string_subclasses_before_callbacks(
    normalizer,
    raw,
):
    """Semantic-control admission must reject hostile text before dispatch."""
    value = _HostileText(raw)

    with pytest.raises(ValueError, match="built-in string"):
        normalizer(value)

    assert value.callbacks == []


@pytest.mark.parametrize(
    "normalizer",
    [backend.normalize_backend, backend.normalize_device],
)
def test_backend_semantic_controls_reject_string_providers_before_callbacks(normalizer):
    """Arbitrary string providers must not be coerced during control admission."""
    value = _HostileStringProvider()

    with pytest.raises(ValueError, match="built-in string"):
        normalizer(value)

    assert value.callbacks == []


def test_backend_semantic_controls_preserve_builtin_normalization():
    """Exact built-in strings retain the documented whitespace/case normalization."""
    assert backend.normalize_backend("  AuTo ") == "auto"
    assert backend.normalize_production_backend(" RUST ") == "rust"
    assert backend.normalize_device(" GPU ") == "gpu"
    with backend._reference_backend_scope():
        assert backend.resolve_reference_backend(" NumPy ") == "numpy"
