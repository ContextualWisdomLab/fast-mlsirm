from __future__ import annotations

import numpy as np

from .config import FitConfig, PenaltyConfig
from .math import sigmoid, softplus
from .types import MLSIRMParams


def prepare_response(responses: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")  # pragma: no cover
    if mask is None:
        observed = np.isfinite(y) & (y != -1)
    else:
        observed = np.asarray(mask, dtype=bool)
        if observed.shape != y.shape:
            raise ValueError("mask shape must match responses")  # pragma: no cover
        observed &= np.isfinite(y) & (y != -1)

    valid_values = y[observed]
    if valid_values.size == 0:
        raise ValueError("responses contain no observed entries")  # pragma: no cover
    if np.any((valid_values != 0) & (valid_values != 1)):
        raise ValueError("observed responses must be 0 or 1")  # pragma: no cover
    if np.any(observed.sum(axis=0) == 0):
        raise ValueError("all-missing item found")  # pragma: no cover
    if np.any(observed.sum(axis=1) == 0):
        raise ValueError("all-missing person found")  # pragma: no cover

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
) -> tuple[np.ndarray, np.ndarray]:
    free_alpha, uses_space = model_flags(model)
    a = params.a if free_alpha else np.ones_like(params.alpha)
    theta_factor = params.theta[:, factor_id]

    if uses_space:
        # Optimized distance computation: replace O(N*J*D) 3D broadcast with O(N*J) 2D dot product
        xi_sq = np.sum(params.xi ** 2, axis=1)
        zeta_sq = np.sum(params.zeta ** 2, axis=1)
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
) -> tuple[float, MLSIRMParams, float]:
    config = config or FitConfig()
    model = config.normalized_model()
    penalty = config.penalty
    y, observed = prepare_response(responses, mask)
    factors = validate_factor_id(factor_id, y.shape[1], params.theta.shape[1])

    if model in {"ULS2PLM", "ULSRM"} and params.theta.shape[1] != 1:
        raise ValueError(f"{model} requires one trait dimension")  # pragma: no cover

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


def _add_penalty(params: MLSIRMParams, penalty: PenaltyConfig, free_alpha: bool, uses_space: bool) -> float:
    value = 0.5 * penalty.lambda_theta * float(np.sum(params.theta * params.theta))
    value += 0.5 * penalty.lambda_b * float(np.sum(params.b * params.b))
    if free_alpha:
        delta = params.alpha - penalty.mu_alpha
        value += 0.5 * penalty.lambda_alpha * float(np.sum(delta * delta))
    if uses_space:
        value += 0.5 * penalty.lambda_xi * float(np.sum(params.xi * params.xi))
        value += 0.5 * penalty.lambda_zeta * float(np.sum(params.zeta * params.zeta))
        value += 0.5 * penalty.lambda_tau * float((params.tau - penalty.mu_tau) ** 2)
    return value
