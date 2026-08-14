"""Failure-path contracts for the secondary Rust extension loaders.

The loaders are small, but they are a trust boundary between Python package
imports and a compiled shared library.  These tests keep missing-entrypoint,
specification, initialization, cleanup, and lock-race behavior explicit.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


SIMPLE_LOADERS = (
    ("fast_mlsirm._ata_core_loader", "ata_core"),
    ("fast_mlsirm._rotation_core_loader", "rotation_core"),
)
LOCKED_LOADERS = (
    ("fast_mlsirm._bifactor_core_loader", "bifactor_core"),
    ("fast_mlsirm._rating_range_core_loader", "rating_range_core"),
)
ALL_LOADERS = SIMPLE_LOADERS + LOCKED_LOADERS


def _set_fake_core(monkeypatch: pytest.MonkeyPatch, binary_path: str | None) -> None:
    """Make the package expose a controlled compiled-core path."""
    package = importlib.import_module("fast_mlsirm")
    fake_core = SimpleNamespace(__file__=binary_path)
    monkeypatch.setattr(package, "_core", fake_core, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake_core)


def _initialize(loader_module: object, public_name: str) -> ModuleType:
    """Invoke either a public-only or a private initializer uniformly."""
    private_initializer = getattr(loader_module, "_initialize_module", None)
    if private_initializer is not None:
        return private_initializer()
    return getattr(loader_module, public_name)()


@pytest.mark.parametrize(("module_name", "public_name"), ALL_LOADERS)
def test_loader_rejects_a_core_without_an_extension_path(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    public_name: str,
) -> None:
    """A source checkout without a compiled extension fails closed."""
    loader_module = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, loader_module._MODULE_NAME, raising=False)
    _set_fake_core(monkeypatch, None)

    with pytest.raises(ImportError, match="extension path"):
        _initialize(loader_module, public_name)


@pytest.mark.parametrize(("module_name", "public_name"), ALL_LOADERS)
def test_loader_rejects_an_unusable_extension_spec(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    public_name: str,
) -> None:
    """A loader that cannot produce a module spec reports an import error."""
    loader_module = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, loader_module._MODULE_NAME, raising=False)
    _set_fake_core(monkeypatch, "/tmp/fast-mlsirm-test-extension.so")

    class DummyLoader:
        """Stand-in that avoids touching a real shared library."""

        def __init__(self, *_args: object) -> None:
            """Accept the standard extension-loader constructor arguments."""

    monkeypatch.setattr(loader_module, "ExtensionFileLoader", DummyLoader)
    monkeypatch.setattr(loader_module, "spec_from_loader", lambda *_args: None)

    with pytest.raises(ImportError, match="could not create"):
        _initialize(loader_module, public_name)


@pytest.mark.parametrize(("module_name", "public_name"), ALL_LOADERS)
def test_loader_removes_partial_module_after_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    public_name: str,
) -> None:
    """A failed PyInit attempt cannot poison later imports with a partial module."""
    loader_module = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, loader_module._MODULE_NAME, raising=False)
    _set_fake_core(monkeypatch, "/tmp/fast-mlsirm-test-extension.so")

    class DummyLoader:
        """Stand-in that deterministically fails module initialization."""

        def __init__(self, *_args: object) -> None:
            """Accept the standard extension-loader constructor arguments."""

        def exec_module(self, _module: ModuleType) -> None:
            """Raise the same class of error a broken PyInit can raise."""
            raise RuntimeError("synthetic PyInit failure")

    monkeypatch.setattr(loader_module, "ExtensionFileLoader", DummyLoader)
    monkeypatch.setattr(loader_module, "spec_from_loader", lambda *_args: object())
    monkeypatch.setattr(
        loader_module,
        "module_from_spec",
        lambda _spec: ModuleType(loader_module._MODULE_NAME),
    )

    with pytest.raises(RuntimeError, match="synthetic PyInit failure"):
        _initialize(loader_module, public_name)
    assert loader_module._MODULE_NAME not in sys.modules


@pytest.mark.parametrize(("module_name", "public_name"), ALL_LOADERS)
def test_loader_publishes_a_successfully_initialized_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    public_name: str,
) -> None:
    """A successful PyInit result is returned and retained for later calls."""
    loader_module = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, loader_module._MODULE_NAME, raising=False)
    _set_fake_core(monkeypatch, "/tmp/fast-mlsirm-test-extension.so")
    initialized = ModuleType(loader_module._MODULE_NAME)

    class DummyLoader:
        """Stand-in that accepts the module without executing native code."""

        def __init__(self, *_args: object) -> None:
            """Accept the standard extension-loader constructor arguments."""

        def exec_module(self, module: ModuleType) -> None:
            """Confirm that the loader receives the module it must initialize."""
            assert module is initialized

    monkeypatch.setattr(loader_module, "ExtensionFileLoader", DummyLoader)
    monkeypatch.setattr(loader_module, "spec_from_loader", lambda *_args: object())
    monkeypatch.setattr(loader_module, "module_from_spec", lambda _spec: initialized)

    try:
        assert getattr(loader_module, public_name)() is initialized
        assert sys.modules[loader_module._MODULE_NAME] is initialized
    finally:
        # The production loader owns this insertion; remove it before the
        # monkeypatch fixture restores the pre-test import state.
        sys.modules.pop(loader_module._MODULE_NAME, None)


@pytest.mark.parametrize(("module_name", "public_name"), ALL_LOADERS)
def test_loader_returns_an_existing_cached_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    public_name: str,
) -> None:
    """Repeated access returns the exact cached module object."""
    loader_module = importlib.import_module(module_name)
    cached = ModuleType(loader_module._MODULE_NAME)
    monkeypatch.setitem(sys.modules, loader_module._MODULE_NAME, cached)

    assert getattr(loader_module, public_name)() is cached


@pytest.mark.parametrize(("module_name", "public_name"), LOCKED_LOADERS)
def test_locked_loader_rechecks_cache_after_acquiring_lock(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    public_name: str,
) -> None:
    """A concurrent initializer that wins the lock is reused, not duplicated."""
    loader_module = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, loader_module._MODULE_NAME, raising=False)
    cached = ModuleType(loader_module._MODULE_NAME)

    class PublishingLock:
        """Publish the competing module while the loader owns the lock."""

        def __enter__(self) -> "PublishingLock":
            """Simulate another importer finishing before the second check."""
            sys.modules[loader_module._MODULE_NAME] = cached
            return self

        def __exit__(self, *_args: object) -> None:
            """Release the synthetic lock without changing the cached module."""

    monkeypatch.setattr(loader_module, "_LOAD_LOCK", PublishingLock())

    try:
        assert getattr(loader_module, public_name)() is cached
    finally:
        # PublishingLock simulates a competing importer, so clean up its
        # production-side sys.modules mutation explicitly.
        sys.modules.pop(loader_module._MODULE_NAME, None)
