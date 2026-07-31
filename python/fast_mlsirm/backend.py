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
    """Lower-case and validate a backend name against ``{numpy, rust, auto}``."""
    backend = str(name).strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    return backend


def normalize_device(name: str) -> str:
    """Lower-case and validate a Rust-backend device against ``{cpu, gpu, auto}``."""
    device = str(name).strip().lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"rust_device must be one of {sorted(VALID_DEVICES)}")
    return device


def resolve_backend(name: str) -> str:
    """Resolve a requested backend to the concrete one that will run.

    ``numpy`` always resolves to ``numpy``; ``rust`` requires the compiled
    ``fast_mlsirm._core`` extension (raising if absent); ``auto`` resolves to
    ``rust`` when the core is importable and to ``numpy`` otherwise.
    """
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
    """Import and return the compiled Rust core, raising if it is unavailable."""
    core = _load_core()
    if core is None:
        raise RuntimeError("Rust backend requested but fast_mlsirm._core is unavailable")
    return core


def _load_core() -> ModuleType | None:
    """Import ``fast_mlsirm._core`` if it is installed, else return ``None``."""
    if importlib.util.find_spec(CORE_MODULE) is None:
        return None
    return importlib.import_module(CORE_MODULE)
