from __future__ import annotations

import importlib

import numpy as np

from .backend import normalize_backend
from .config import FitConfig, PenaltyConfig
from .math import sigmoid, softplus
from .types import MLSIRMParams


def prepare_response(responses: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")
    if mask is None:
        observed = np.isfinite(y) & (y != -1)
    else:
        observed = np.asarray(mask, dtype=bool)
        if observed.shape != y.shape:
            raise ValueError("mask shape must match responses")
        observed &= np.isfinite(y) & (y != -1)

    if not np.any(observed):
        raise ValueError("responses contain no observed entries")

    invalid = (y != 0) & (y != 1)
    if np.any(observed & invalid):
        raise ValueError("observed responses must be 0 or 1")

    if np.any(observed.sum(axis=0) == 0):
        raise ValueError("all-missing item found")
    if np.any(observed.sum(axis=1) == 0):
        raise ValueError("all-missing person found")

    clean = np.where(observed, y, 0.0)
    return clean, observed


def validate_factor_id(factor_id: np.ndarray, n_items: int, n_dims: int) -> np.ndarray:
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != (n_items,):
        raise ValueError("factor_id length must match number of items")
    if np.any(factors < 0) or np.any(factors >= n_dims):
        raise ValueError("factor_id values must be in 0..n_dims-1")
    return factors


def model_flags(model: str) -> tuple[bool, bool]:
    name = model.upper()
    free_alpha = name not in {"MLSRM", "ULSRM"}
    uses_space = name != "MIRT"
    return free_alpha, uses_space


def linear_predictor(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    model: str = "MLS2PLM",
    eps_distance: float = 1e-8,
    compute_backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    backend = normalize_backend(compute_backend)
    if backend != "cpu":
        return _linear_predictor_backend(params, factor_id, model=model, eps_distance=eps_distance, backend=backend)

    free_alpha, uses_space = model_flags(model)
    a = params.a if free_alpha else np.ones_like(params.alpha)
    theta_factor = params.theta[:, factor_id]

    if uses_space:
        # Optimized distance computation: replace O(N*J*D) 3D broadcast with O(N*J) 2D dot product
        xi_sq = np.einsum('ij,ij->i', params.xi, params.xi)
        zeta_sq = np.einsum('ij,ij->i', params.zeta, params.zeta)
        dist_sq = xi_sq[:, None] + zeta_sq[None, :] - 2 * np.dot(params.xi, params.zeta.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        distance = np.sqrt(dist_sq + eps_distance)
        gamma = params.gamma
    else:
        distance = np.zeros((params.theta.shape[0], len(factor_id)), dtype=np.float64)
        gamma = 0.0

    eta = a[None, :] * theta_factor + params.b[None, :] - gamma * distance
    return eta, distance


def neg_loglik_and_grad(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    compute_backend: str = "cpu",
) -> tuple[float, MLSIRMParams, float]:
    config = config or FitConfig()
    backend = normalize_backend(compute_backend)
    if backend != "cpu":
        return _neg_loglik_and_grad_backend(responses, factor_id, params, config, mask, backend)

    model = config.normalized_model()
    penalty = config.penalty
    y, observed = prepare_response(responses, mask)
    factors = validate_factor_id(factor_id, y.shape[1], params.theta.shape[1])

    if model in {"ULS2PLM", "ULSRM"} and params.theta.shape[1] != 1:
        raise ValueError(f"{model} requires one trait dimension")

    free_alpha, uses_space = model_flags(model)
    a = params.a if free_alpha else np.ones_like(params.alpha)
    eta, distance = linear_predictor(params, factors, model=model, eps_distance=config.eps_distance)
    pi = sigmoid(eta)
    entry_loss = (softplus(eta) - y * eta) * observed
    nll = float(entry_loss.sum())
    loglik = -nll

    e = (pi - y) * observed
    grad_b = e.sum(axis=0)
    grad_alpha = np.zeros_like(params.alpha)
    if free_alpha:
        grad_alpha = (e * a[None, :] * params.theta[:, factors]).sum(axis=0)

    # Optimized gradient computation: replace loop over dimensions with matrix multiplication
    # np.eye(...)[factors] creates a one-hot encoding (J x D), projecting J items onto D dimensions
    I = np.zeros((e.shape[1], params.theta.shape[1]), dtype=e.dtype)
    I[np.arange(e.shape[1]), factors] = 1
    grad_theta = (e * a[None, :]) @ I

    grad_xi = np.zeros_like(params.xi)
    grad_zeta = np.zeros_like(params.zeta)
    grad_tau = 0.0
    if uses_space:
        gamma = params.gamma

        # Optimized gradient computation: avoid 3D array creation, use 2D matrix multiplication instead
        e_over_d = e / distance
        sum_e_over_d = e_over_d.sum(axis=1, keepdims=True)
        grad_xi = -gamma * (params.xi * sum_e_over_d - np.dot(e_over_d, params.zeta))

        sum_e_over_d_j = e_over_d.sum(axis=0, keepdims=True).T
        grad_zeta = gamma * (np.dot(e_over_d.T, params.xi) - params.zeta * sum_e_over_d_j)

        grad_tau = float((e * (-gamma * distance)).sum())

    nll += _add_penalty(params, penalty, free_alpha=free_alpha, uses_space=uses_space)
    grad_theta += penalty.lambda_theta * params.theta
    grad_b += penalty.lambda_b * params.b
    if free_alpha:
        grad_alpha += penalty.lambda_alpha * (params.alpha - penalty.mu_alpha)
    if uses_space:
        grad_xi += penalty.lambda_xi * params.xi
        grad_zeta += penalty.lambda_zeta * params.zeta
        grad_tau += penalty.lambda_tau * (params.tau - penalty.mu_tau)

    grads = MLSIRMParams(
        theta=grad_theta,
        alpha=grad_alpha,
        b=grad_b,
        xi=grad_xi,
        zeta=grad_zeta,
        tau=float(grad_tau),
    )
    return float(nll), grads, loglik


def _linear_predictor_backend(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    model: str,
    eps_distance: float,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    xp, to_numpy = _backend_module(backend)
    free_alpha, uses_space = model_flags(model)
    factors = xp.asarray(np.asarray(factor_id, dtype=np.int64))
    theta = xp.asarray(params.theta)
    alpha = xp.asarray(params.alpha)
    b = xp.asarray(params.b)
    xi = xp.asarray(params.xi)
    zeta = xp.asarray(params.zeta)
    a = xp.exp(alpha) if free_alpha else xp.ones_like(alpha)
    theta_factor = theta[:, factors]

    if uses_space:
        xi_sq = xp.sum(xi * xi, axis=1)
        zeta_sq = xp.sum(zeta * zeta, axis=1)
        dist_sq = xi_sq[:, None] + zeta_sq[None, :] - 2.0 * (xi @ zeta.T)
        dist_sq = xp.maximum(dist_sq, 0.0)
        distance = xp.sqrt(dist_sq + eps_distance)
        gamma = float(np.exp(params.tau))
    else:
        distance = xp.zeros((theta.shape[0], len(factor_id)))
        gamma = 0.0

    eta = a[None, :] * theta_factor + b[None, :] - gamma * distance
    return to_numpy(eta), to_numpy(distance)


def _neg_loglik_and_grad_backend(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig,
    mask: np.ndarray | None,
    backend: str,
) -> tuple[float, MLSIRMParams, float]:
    model = config.normalized_model()
    penalty = config.penalty
    y, observed = prepare_response(responses, mask)
    factors_np = validate_factor_id(factor_id, y.shape[1], params.theta.shape[1])
    if model in {"ULS2PLM", "ULSRM"} and params.theta.shape[1] != 1:
        raise ValueError(f"{model} requires one trait dimension")

    xp, to_numpy = _backend_module(backend)
    y_dev = xp.asarray(y)
    observed_dev = xp.asarray(observed)
    factors = xp.asarray(factors_np)
    theta = xp.asarray(params.theta)
    alpha = xp.asarray(params.alpha)
    b = xp.asarray(params.b)
    xi = xp.asarray(params.xi)
    zeta = xp.asarray(params.zeta)

    free_alpha, uses_space = model_flags(model)
    a = xp.exp(alpha) if free_alpha else xp.ones_like(alpha)
    theta_factor = theta[:, factors]

    if uses_space:
        xi_sq = xp.sum(xi * xi, axis=1)
        zeta_sq = xp.sum(zeta * zeta, axis=1)
        dist_sq = xi_sq[:, None] + zeta_sq[None, :] - 2.0 * (xi @ zeta.T)
        dist_sq = xp.maximum(dist_sq, 0.0)
        distance = xp.sqrt(dist_sq + config.eps_distance)
        gamma = float(np.exp(params.tau))
    else:
        distance = xp.zeros((theta.shape[0], y.shape[1]))
        gamma = 0.0

    eta = a[None, :] * theta_factor + b[None, :] - gamma * distance
    eta_safe = xp.clip(eta, -709.0, 709.0)
    pi = 1.0 / (1.0 + xp.exp(-eta_safe))
    entry_loss = (xp.maximum(eta, 0.0) + xp.log1p(xp.exp(-xp.abs(eta))) - y_dev * eta) * observed_dev
    nll = float(np.asarray(to_numpy(xp.sum(entry_loss))))
    loglik = -nll

    e = (pi - y_dev) * observed_dev
    grad_b = xp.sum(e, axis=0)
    grad_alpha = xp.zeros_like(alpha)
    if free_alpha:
        grad_alpha = xp.sum(e * a[None, :] * theta[:, factors], axis=0)

    grad_theta = xp.zeros_like(theta)
    for d in range(theta.shape[1]):
        mask_d = factors == d
        if float(np.asarray(to_numpy(xp.sum(mask_d)))) > 0.0:
            grad_theta[:, d] = xp.sum(e[:, mask_d] * a[None, mask_d], axis=1)

    grad_xi = xp.zeros_like(xi)
    grad_zeta = xp.zeros_like(zeta)
    grad_tau = 0.0
    if uses_space:
        e_over_d = e / distance
        sum_e_over_d = xp.sum(e_over_d, axis=1, keepdims=True)
        grad_xi = -gamma * (xi * sum_e_over_d - (e_over_d @ zeta))
        sum_e_over_d_j = xp.sum(e_over_d, axis=0)[:, None]
        grad_zeta = gamma * ((e_over_d.T @ xi) - zeta * sum_e_over_d_j)
        grad_tau = float(np.asarray(to_numpy(xp.sum(e * (-gamma * distance)))))

    nll += _add_penalty(params, penalty, free_alpha=free_alpha, uses_space=uses_space)
    grad_theta += penalty.lambda_theta * theta
    grad_b += penalty.lambda_b * b
    if free_alpha:
        grad_alpha += penalty.lambda_alpha * (alpha - penalty.mu_alpha)
    if uses_space:
        grad_xi += penalty.lambda_xi * xi
        grad_zeta += penalty.lambda_zeta * zeta
        grad_tau += penalty.lambda_tau * (params.tau - penalty.mu_tau)

    grads = MLSIRMParams(
        theta=np.asarray(to_numpy(grad_theta)),
        alpha=np.asarray(to_numpy(grad_alpha)),
        b=np.asarray(to_numpy(grad_b)),
        xi=np.asarray(to_numpy(grad_xi)),
        zeta=np.asarray(to_numpy(grad_zeta)),
        tau=float(grad_tau),
    )
    return float(nll), grads, loglik


def _backend_module(backend: str):
    """Return the array module and a converter that materializes arrays to NumPy."""
    if backend == "cuda":
        cp = importlib.import_module("cupy")
        return cp, cp.asnumpy
    if backend == "mlx":
        mx = importlib.import_module("mlx.core")
        return mx, _mlx_to_numpy
    if backend == "opencl":
        # OpenCL backend runs in compatibility mode (platform/device validated at fit start)
        # to preserve formula parity until dedicated OpenCL kernels are added.
        return np, np.asarray
    return np, np.asarray


def _mlx_to_numpy(value):
    # `np.asarray` triggers MLX array conversion through its numpy interop protocol.
    return np.asarray(value)


def _add_penalty(params: MLSIRMParams, penalty: PenaltyConfig, free_alpha: bool, uses_space: bool) -> float:
    # Optimized penalty calculation: replace np.sum(x * x) with np.vdot(x, x) to avoid intermediate array allocation
    value = 0.5 * penalty.lambda_theta * float(np.vdot(params.theta, params.theta))
    value += 0.5 * penalty.lambda_b * float(np.vdot(params.b, params.b))
    if free_alpha:
        delta = params.alpha - penalty.mu_alpha
        value += 0.5 * penalty.lambda_alpha * float(np.vdot(delta, delta))
    if uses_space:
        value += 0.5 * penalty.lambda_xi * float(np.vdot(params.xi, params.xi))
        value += 0.5 * penalty.lambda_zeta * float(np.vdot(params.zeta, params.zeta))
        value += 0.5 * penalty.lambda_tau * float((params.tau - penalty.mu_tau) ** 2)
    return value
