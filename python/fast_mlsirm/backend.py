from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


VALID_BACKENDS = {"numpy", "rust", "auto"}
# Execution device for the Rust backend. This is a sub-option of the ``rust``
# backend (CPU vs. wgpu GPGPU), NOT a separate compute-backend axis: the single
# backend axis stays {numpy, rust, auto}. ``auto``/``gpu`` run the GPGPU kernels
# when a GPU is present and fall back to the identical CPU path otherwise.
VALID_DEVICES = {"cpu", "gpu", "auto"}
CORE_MODULE = "fast_mlsirm._core"


def normalize_backend(name: str) -> str:
    backend = str(name).strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    return backend


def normalize_device(name: str) -> str:
    device = str(name).strip().lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"rust_device must be one of {sorted(VALID_DEVICES)}")
    return device


def resolve_backend(name: str) -> str:
    backend = normalize_backend(name)
    if backend == "numpy":
        return "numpy"
    core = _load_core()
    if backend == "rust":
        if core is None:
            raise RuntimeError("Rust backend requested but fast_mlsirm._core is unavailable")
        return "rust"
    return "rust" if core is not None else "numpy"


def load_rust_core() -> ModuleType:
    core = _load_core()
    if core is None:
        raise RuntimeError("Rust backend requested but fast_mlsirm._core is unavailable")
    return core


def _load_core() -> ModuleType | None:
    if importlib.util.find_spec(CORE_MODULE) is None:
        return None
    return importlib.import_module(CORE_MODULE)
