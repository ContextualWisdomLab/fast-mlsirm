from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


VALID_BACKENDS = {"numpy", "rust", "auto"}
# Execution device for the Rust backend. This is a sub-option of the ``rust``
# backend (CPU vs. wgpu GPGPU), NOT a separate compute-backend axis: the single
# backend axis stays {numpy, rust, auto}. ``auto``/``gpu`` run the GPGPU kernels
# when a GPU is present and fall back to the identical Rust CPU path otherwise.
VALID_DEVICES = {"cpu", "gpu", "auto"}
CORE_MODULE = "fast_mlsirm._core"
# Stable, non-reflective auto-resolution error. Do not interpolate paths,
# ABI details, environment data, or import exception text into this text.
AUTO_BACKEND_UNAVAILABLE_MESSAGE = (
    "compiled Rust core is required for automatic backend resolution; "
    "install a wheel or editable build that provides fast_mlsirm._core, "
    "or pass backend='numpy' only for the explicit reference/parity path"
)


def normalize_backend(name: str) -> str:
    """Lower-case and validate an exact built-in backend control string."""
    if type(name) is not str:
        raise ValueError("backend must be an exact built-in string")
    backend = name.strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    return backend


def normalize_device(name: str) -> str:
    """Lower-case and validate an exact built-in Rust-device control string."""
    if type(name) is not str:
        raise ValueError("rust_device must be an exact built-in string")
    device = name.strip().lower()
    if device not in VALID_DEVICES:
        raise ValueError(f"rust_device must be one of {sorted(VALID_DEVICES)}")
    return device


def resolve_backend(name: str) -> str:
    """Resolve a requested numerical backend to the concrete backend that runs.

    ``numpy`` is an explicit reference/parity choice and always resolves to
    ``numpy``. ``rust`` requires the compiled :mod:`fast_mlsirm._core`
    extension. ``auto`` is the production convenience choice: it resolves to
    ``rust`` when the compiled core is available and fails closed otherwise.
    Automatic resolution never silently changes the numerical owner to NumPy.
    """
    backend = normalize_backend(name)
    if backend == "numpy":
        return "numpy"
    core = _load_core()
    if backend == "rust":
        if core is None:
            raise RuntimeError("Rust backend requested but fast_mlsirm._core is unavailable")
        return "rust"
    if core is None:
        raise RuntimeError(AUTO_BACKEND_UNAVAILABLE_MESSAGE)
    return "rust"


def load_rust_core() -> ModuleType:
    """Import and return the compiled Rust core, raising if it is unavailable."""
    core = _load_core()
    if core is None:
        raise RuntimeError("Rust backend requested but fast_mlsirm._core is unavailable")
    return core


def _load_core() -> ModuleType | None:
    """Import the Rust core or normalize native loader failures.

    A missing extension is represented as ``None`` so the public resolver can
    retain its existing missing-core messages. If discovery succeeds but the
    native module cannot be loaded, fail closed with a package-owned error and
    preserve the loader exception as the cause for operator diagnostics.
    """
    if importlib.util.find_spec(CORE_MODULE) is None:
        return None
    try:
        return importlib.import_module(CORE_MODULE)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "compiled Rust core is present but could not be imported"
        ) from exc
