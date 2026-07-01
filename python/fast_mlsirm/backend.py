from __future__ import annotations

import importlib


VALID_COMPUTE_BACKENDS = {"cpu", "cuda", "mlx", "opencl"}


def normalize_backend(name: str) -> str:
    backend = str(name).strip().lower()
    if backend not in VALID_COMPUTE_BACKENDS:
        raise ValueError(f"compute_backend must be one of {sorted(VALID_COMPUTE_BACKENDS)}")
    return backend


def ensure_backend_available(name: str) -> str:
    backend = normalize_backend(name)
    if backend == "cpu":
        return backend
    if backend == "cuda":
        _require_module("cupy", "CUDA backend requires cupy.")
        return backend
    if backend == "mlx":
        _require_module("mlx.core", "MLX backend requires mlx.")
        return backend
    _require_opencl()
    return backend


def _require_module(module_name: str, message: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ValueError(message) from exc


def _require_opencl() -> None:
    try:
        import pyopencl as cl
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ValueError("OpenCL backend requires pyopencl.") from exc

    try:
        platforms = cl.get_platforms()
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise ValueError("OpenCL backend is unavailable because no OpenCL platform was found.") from exc

    if not platforms:
        raise ValueError("OpenCL backend is unavailable because no OpenCL platform was found.")
