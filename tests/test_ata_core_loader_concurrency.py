"""Concurrency regression tests for the secondary ATA extension loader."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
import sys
import threading
import time

import fast_mlsirm
import fast_mlsirm._ata_core_loader as loader


def test_ata_core_does_not_return_partially_initialized_module(monkeypatch):
    """Concurrent callers must wait until the secondary module finishes initialization."""
    target = loader._MODULE_NAME
    sys.modules.pop(target, None)

    fake_core = ModuleType("fast_mlsirm._core")
    fake_core.__file__ = "/tmp/fake-fast-mlsirm-core.so"
    monkeypatch.setattr(fast_mlsirm, "_core", fake_core)

    exec_started = threading.Event()
    release_exec = threading.Event()
    exec_calls = 0

    class FakeExtensionLoader:
        """Expose the publication window without loading a native library."""

        def __init__(self, name: str, path: str) -> None:
            self.name = name
            self.path = path

        def exec_module(self, module: ModuleType) -> None:
            nonlocal exec_calls
            exec_calls += 1
            module.initialized = False
            exec_started.set()
            if not release_exec.wait(timeout=2.0):
                raise AssertionError("test did not release simulated extension initialization")
            module.initialized = True

    monkeypatch.setattr(loader, "ExtensionFileLoader", FakeExtensionLoader)
    monkeypatch.setattr(loader, "spec_from_loader", lambda _name, _loader: object())
    monkeypatch.setattr(loader, "module_from_spec", lambda _spec: ModuleType(target))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(loader.ata_core)
            assert exec_started.wait(timeout=1.0)
            second = pool.submit(loader.ata_core)
            time.sleep(0.05)
            assert not second.done(), (
                "ata_core returned the sys.modules entry while the first "
                "caller was still initializing it"
            )
            release_exec.set()
            first_module = first.result(timeout=1.0)
            second_module = second.result(timeout=1.0)
    finally:
        release_exec.set()
        sys.modules.pop(target, None)

    assert first_module is second_module
    assert first_module.initialized is True
    assert exec_calls == 1
