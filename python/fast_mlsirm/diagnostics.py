from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from .config import FitConfig
from .math import sigmoid, standardize
from .objective import linear_predictor, model_flags, prepare_response
from .types import DimensionalityDiagnostics, FitDiagnostics, MLSIRMParams, RecoveryReport


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


def fit_diagnostics(
    responses: np.ndarray,
    params: MLSIRMParams,
    factor_id: np.ndarray,
    mask: np.ndarray | None = None,
    model: str = "MLS2PLM",
    parameter_count: int | None = None,
    eps: float = 1e-12,
) -> FitDiagnostics:
    y, observed = prepare_response(responses, mask)
    prob = np.clip(predict_proba(params, factor_id, model=model), eps, 1.0 - eps)
    if prob.shape != y.shape:
        raise ValueError("parameter dimensions must match responses and factor_id")

    variance = np.maximum(prob * (1.0 - prob), eps)
    residual = (y - prob) * observed
    pearson_sq = np.where(observed, residual * residual / variance, 0.0)
    n_parameters = int(parameter_count) if parameter_count is not None else _parameter_count(params, model)

    itemfit = _axis_fit(y, observed, prob, variance, residual, pearson_sq, axis=0)
    personfit = _axis_fit(y, observed, prob, variance, residual, pearson_sq, axis=1)
    factorfit = _factor_fit(factor_id, y, observed, prob, variance, residual, pearson_sq)
    loglik = float(np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0).sum())
    n_observed = int(observed.sum())
    deviance = -2.0 * loglik
    model_fit = {
        "loglik": loglik,
        "deviance": deviance,
        "aic": 2.0 * n_parameters - 2.0 * loglik,
        "bic": np.log(n_observed) * n_parameters - 2.0 * loglik,
        "n_observed": float(n_observed),
        "parameter_count": float(n_parameters),
        "observed_mean": float(y[observed].mean()),
        "expected_mean": float(prob[observed].mean()),
        "mean_abs_residual": float(np.abs(residual[observed]).mean()),
        "pearson_chisq": float(pearson_sq.sum()),
    }
    return FitDiagnostics(itemfit=itemfit, personfit=personfit, model_fit=model_fit, factorfit=factorfit)


def dimensionality_diagnostics(
    responses: np.ndarray,
    factor_id: np.ndarray,
    latent_dims: Iterable[int],
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    model: str = "MLS2PLM",
    k_folds: int = 5,
    seed: int = 1,
    eps: float = 1e-12,
) -> DimensionalityDiagnostics:
    from .fit import fit

    y, observed = prepare_response(responses, mask)
    folds = _validation_folds(observed, k_folds, seed)
    base = config or FitConfig(model=model)
    candidates: list[dict[str, float]] = []

    for latent_dim in _validated_latent_dims(latent_dims):
        totals = {"loglik": 0.0, "abs_residual": 0.0, "sq_residual": 0.0, "n": 0.0}
        for fold_idx, validation_mask in enumerate(folds):
            train_mask = observed & ~validation_mask
            fitted = fit(
                y,
                factor_id,
                config=replace(base, model=model, latent_dim=latent_dim, seed=seed + fold_idx),
                mask=train_mask,
            )
            prob = np.clip(predict_proba(fitted.params, factor_id, model=fitted.model), eps, 1.0 - eps)
            _accumulate_heldout(totals, y, validation_mask, prob)

        candidates.append(
            {
                "latent_dim": float(latent_dim),
                "k_folds": float(k_folds),
                "heldout_loglik": totals["loglik"],
                "heldout_deviance": -2.0 * totals["loglik"],
                "heldout_mean_abs_residual": totals["abs_residual"] / totals["n"],
                "heldout_rmse": float(np.sqrt(totals["sq_residual"] / totals["n"])),
                "n_heldout": totals["n"],
            }
        )

    best = max(candidates, key=lambda row: row["heldout_loglik"])
    return DimensionalityDiagnostics(candidates=candidates, best=best)


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


def _axis_fit(
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
    axis: int,
) -> dict[str, np.ndarray]:
    count = observed.sum(axis=axis).astype(np.float64)
    score = (y * observed).sum(axis=axis)
    expected = (prob * observed).sum(axis=axis)
    raw = residual.sum(axis=axis)
    variance_sum = (variance * observed).sum(axis=axis)
    return {
        "observed_count": count,
        "score": score,
        "expected_score": expected,
        "raw_residual": raw,
        "standardized_residual": raw / np.sqrt(variance_sum),
        "infit_mnsq": (residual * residual).sum(axis=axis) / variance_sum,
        "outfit_mnsq": pearson_sq.sum(axis=axis) / count,
    }


def _factor_fit(
    factor_id: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
) -> dict[str, np.ndarray]:
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != (y.shape[1],):
        raise ValueError("factor_id length must match number of items")

    rows = []
    for factor in np.unique(factors):
        cols = factors == factor
        rows.append(
            (
                float(factor),
                float(observed[:, cols].sum()),
                float((y[:, cols] * observed[:, cols]).sum()),
                float((prob[:, cols] * observed[:, cols]).sum()),
                float(residual[:, cols].sum()),
                float((variance[:, cols] * observed[:, cols]).sum()),
                float((residual[:, cols] * residual[:, cols]).sum()),
                float(pearson_sq[:, cols].sum()),
            )
        )

    table = np.asarray(rows, dtype=np.float64)
    variance_sum = table[:, 5]
    count = table[:, 1]
    return {
        "factor_id": table[:, 0],
        "observed_count": count,
        "score": table[:, 2],
        "expected_score": table[:, 3],
        "raw_residual": table[:, 4],
        "standardized_residual": table[:, 4] / np.sqrt(variance_sum),
        "infit_mnsq": table[:, 6] / variance_sum,
        "outfit_mnsq": table[:, 7] / count,
    }


def _parameter_count(params: MLSIRMParams, model: str) -> int:
    free_alpha, uses_space = model_flags(model)
    count = params.theta.size + params.b.size
    if free_alpha:
        count += params.alpha.size
    if uses_space:
        count += params.xi.size + params.zeta.size + 1
    return count


def _validated_latent_dims(latent_dims: Iterable[int]) -> list[int]:
    dims = [int(value) for value in latent_dims]
    if not dims:
        raise ValueError("latent_dims must not be empty")
    if any(value < 1 for value in dims):
        raise ValueError("latent_dims must be >= 1")
    return dims


def _validation_folds(observed: np.ndarray, k_folds: int, seed: int) -> list[np.ndarray]:
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")

    row_counts = observed.sum(axis=1)
    col_counts = observed.sum(axis=0)
    eligible = np.argwhere(observed & (row_counts[:, None] > 1) & (col_counts[None, :] > 1))
    if len(eligible) < k_folds:
        raise ValueError("not enough observed entries for k-fold validation")

    rng = np.random.default_rng(seed)
    splits = np.array_split(rng.permutation(len(eligible)), k_folds)
    folds: list[np.ndarray] = []
    for split in splits:
        mask = np.zeros_like(observed, dtype=bool)
        rows = eligible[split, 0]
        cols = eligible[split, 1]
        mask[rows, cols] = True
        train = observed & ~mask
        mask[train.sum(axis=1) == 0, :] = False
        mask[:, train.sum(axis=0) == 0] = False
        if not np.any(mask):
            raise ValueError("fold validation set is empty; reduce k_folds")
        folds.append(mask)
    return folds


def _accumulate_heldout(totals: dict[str, float], y: np.ndarray, mask: np.ndarray, prob: np.ndarray) -> None:
    yy = y[mask]
    pp = prob[mask]
    residual = yy - pp
    totals["loglik"] += float((yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp)).sum())
    totals["abs_residual"] += float(np.abs(residual).sum())
    totals["sq_residual"] += float((residual * residual).sum())
    totals["n"] += float(mask.sum())


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
