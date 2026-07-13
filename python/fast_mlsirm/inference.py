from __future__ import annotations

import numpy as np

from .config import FitConfig
from .fit import _pack, _unpack
from .objective import neg_loglik_and_grad
from .types import MLSIRMParams


def observed_information(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    backend: str | None = None,
    device: str | None = "cpu",
    step: float = 1e-4,
) -> np.ndarray:
    """Finite-difference Hessian of the penalized negative log-likelihood.

    The default Rust device is CPU so finite-difference curvature uses the f64
    path even when model fitting defaults to ``rust_device="auto"`` on GPU hosts.
    Pass ``device=None`` to honor ``config.rust_device`` instead.
    """
    config = config or FitConfig()
    model = config.normalized_model()
    chosen_backend = config.backend if backend is None else backend
    x0 = _pack(params, model)
    if step <= 0:
        raise ValueError("step must be > 0")

    def objective(x: np.ndarray) -> float:
        value, _, _ = neg_loglik_and_grad(
            responses,
            factor_id,
            _unpack(x, params, model),
            config=config,
            mask=mask,
            backend=chosen_backend,
            device=device,
        )
        if not np.isfinite(value):
            raise ValueError("objective must be finite for Hessian calculation")
        return float(value)

    n = x0.size
    hessian = np.zeros((n, n), dtype=np.float64)
    base = objective(x0)
    eye = np.eye(n, dtype=np.float64)
    h = float(step)

    for i in range(n):
        x_plus = x0 + h * eye[i]
        x_minus = x0 - h * eye[i]
        hessian[i, i] = (objective(x_plus) - 2.0 * base + objective(x_minus)) / (h * h)
        for j in range(i + 1, n):
            f_pp = objective(x0 + h * eye[i] + h * eye[j])
            f_pm = objective(x0 + h * eye[i] - h * eye[j])
            f_mp = objective(x0 - h * eye[i] + h * eye[j])
            f_mm = objective(x0 - h * eye[i] - h * eye[j])
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            hessian[i, j] = value
            hessian[j, i] = value

    return (hessian + hessian.T) / 2.0


def second_order_test(hessian: np.ndarray, tol: float = 1e-8) -> dict[str, float | bool | np.ndarray]:
    """Check whether the Hessian/information matrix is positive definite."""
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hessian must be a square matrix")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    min_eigenvalue = float(eigenvalues.min()) if eigenvalues.size else float("nan")
    return {
        "passed": bool(np.all(eigenvalues > tol)),
        "min_eigenvalue": min_eigenvalue,
        "eigenvalues": eigenvalues,
    }


def vcov_from_hessian(hessian: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Invert the observed information, falling back to a Moore-Penrose inverse."""
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hessian must be a square matrix")
    try:
        vcov = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(matrix, rcond=rcond)
    return (vcov + vcov.T) / 2.0


def standard_errors_from_vcov(vcov: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vcov, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("vcov must be a square matrix")
    return np.sqrt(np.maximum(np.diag(matrix), 0.0))
