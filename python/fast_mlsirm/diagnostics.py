from __future__ import annotations

import numpy as np

from .math import sigmoid, standardize
from .objective import linear_predictor
from .types import MLSIRMParams, RecoveryReport


def predict_proba(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    persons: np.ndarray | None = None,
    items: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    sub = _subset_params(params, persons, items)
    factors = np.asarray(factor_id, dtype=np.int64)
    if items is not None:
        factors = factors[np.asarray(items, dtype=np.int64)]
    eta, _ = linear_predictor(sub, factors, model=model)
    return sigmoid(eta)


def align_latent_space(
    true_xi: np.ndarray,
    true_zeta: np.ndarray,
    est_xi: np.ndarray,
    est_zeta: np.ndarray,
    method: str = "procrustes",
) -> tuple[np.ndarray, np.ndarray]:
    if method != "procrustes":
        raise ValueError("only procrustes alignment is supported")

    true = np.vstack([true_xi, true_zeta]).astype(np.float64)
    est = np.vstack([est_xi, est_zeta]).astype(np.float64)
    true_mean = true.mean(axis=0)
    est_mean = est.mean(axis=0)
    true_c = true - true_mean
    est_c = est - est_mean

    u, s, vt = np.linalg.svd(est_c.T @ true_c, full_matrices=False)
    rotation = u @ vt
    denom = float(np.sum(est_c * est_c))
    scale = float(np.sum(s) / denom) if denom > 1e-12 else 1.0
    aligned = scale * est_c @ rotation + true_mean
    return aligned[: len(true_xi)], aligned[len(true_xi) :]


def recovery_report(truth: MLSIRMParams, estimate: MLSIRMParams, align: bool = True) -> RecoveryReport:
    est_xi = estimate.xi
    est_zeta = estimate.zeta
    if align:
        est_xi, est_zeta = align_latent_space(truth.xi, truth.zeta, estimate.xi, estimate.zeta)

    metrics = {
        "a_bias": _bias(truth.a, estimate.a),
        "a_rmse": _rmse(truth.a, estimate.a),
        "a_corr": _corr(truth.a, estimate.a),
        "b_bias": _bias(truth.b, estimate.b),
        "b_rmse": _rmse(truth.b, estimate.b),
        "b_corr": _corr(truth.b, estimate.b),
        "gamma_abs_error": float(abs(truth.gamma - estimate.gamma)),
        "gamma_relative_error": float(abs(truth.gamma - estimate.gamma) / max(abs(truth.gamma), 1e-12)),
        "theta_rmse_standardized": _rmse(standardize(truth.theta), standardize(estimate.theta)),
        "latent_coordinate_rmse": _rmse(np.vstack([truth.xi, truth.zeta]), np.vstack([est_xi, est_zeta])),
        "person_item_distance_rmse": _distance_rmse(truth.xi, truth.zeta, estimate.xi, estimate.zeta),
    }
    summary = {
        "parameter_rmse_mean": float(np.nanmean([metrics["a_rmse"], metrics["b_rmse"], metrics["theta_rmse_standardized"]])),
        "latent_rmse": metrics["latent_coordinate_rmse"],
        "distance_rmse": metrics["person_item_distance_rmse"],
        "gamma_abs_error": metrics["gamma_abs_error"],
    }
    return RecoveryReport(summary=summary, metrics=metrics)


def _subset_params(params: MLSIRMParams, persons: np.ndarray | None, items: np.ndarray | None) -> MLSIRMParams:
    p_idx = slice(None) if persons is None else np.asarray(persons, dtype=np.int64)
    i_idx = slice(None) if items is None else np.asarray(items, dtype=np.int64)
    return MLSIRMParams(
        theta=params.theta[p_idx],
        alpha=params.alpha[i_idx],
        b=params.b[i_idx],
        xi=params.xi[p_idx],
        zeta=params.zeta[i_idx],
        tau=params.tau,
    )


def _bias(true: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.asarray(estimate) - np.asarray(true)))


def _rmse(true: np.ndarray, estimate: np.ndarray) -> float:
    delta = np.asarray(estimate) - np.asarray(true)
    return float(np.sqrt(np.mean(delta * delta)))


def _corr(true: np.ndarray, estimate: np.ndarray) -> float:
    x = np.asarray(true).ravel()
    y = np.asarray(estimate).ravel()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")  # pragma: no cover
    return float(np.corrcoef(x, y)[0, 1])


def _distance_rmse(true_xi: np.ndarray, true_zeta: np.ndarray, est_xi: np.ndarray, est_zeta: np.ndarray) -> float:
    true_d = np.sqrt(((true_xi[:, None, :] - true_zeta[None, :, :]) ** 2).sum(axis=2))
    est_d = np.sqrt(((est_xi[:, None, :] - est_zeta[None, :, :]) ** 2).sum(axis=2))
    return _rmse(true_d, est_d)
