from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from .config import FitConfig
from .math import sigmoid, standardize
from .objective import linear_predictor, model_flags, prepare_response
from .types import (
    DimensionalityDiagnostics,
    FitDiagnostics,
    MLSIRMParams,
    RecoveryReport,
)


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
    *,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
) -> FitDiagnostics:
    y, observed = prepare_response(responses, mask)
    prob = np.clip(predict_proba(params, factor_id, model=model), eps, 1.0 - eps)
    if prob.shape != y.shape:
        raise ValueError("parameter dimensions must match responses and factor_id")

    variance = np.maximum(prob * (1.0 - prob), eps)
    residual = (y - prob) * observed
    pearson_sq = np.where(observed, residual * residual / variance, 0.0)
    n_parameters = (
        int(parameter_count)
        if parameter_count is not None
        else _parameter_count(params, model)
    )

    itemfit = _axis_fit(y, observed, prob, variance, residual, pearson_sq, axis=0)
    personfit = _axis_fit(y, observed, prob, variance, residual, pearson_sq, axis=1)
    _attach_person_strata(
        personfit, group_id=group_id, cluster_id=cluster_id, n_persons=y.shape[0]
    )
    factorfit = _factor_fit(
        factor_id, y, observed, prob, variance, residual, pearson_sq
    )
    loglik = float(
        np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0).sum()
    )
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
    return FitDiagnostics(
        itemfit=itemfit,
        personfit=personfit,
        model_fit=model_fit,
        factorfit=factorfit,
        groupfit=_binary_stratum_fit(
            "group_id", group_id, y, observed, prob, variance, residual, pearson_sq
        ),
        clusterfit=_binary_stratum_fit(
            "cluster_id", cluster_id, y, observed, prob, variance, residual, pearson_sq
        ),
        group_itemfit=_binary_stratum_item_fit(
            "group_id", group_id, y, observed, prob, variance, residual, pearson_sq
        ),
        cluster_itemfit=_binary_stratum_item_fit(
            "cluster_id", cluster_id, y, observed, prob, variance, residual, pearson_sq
        ),
    )


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
                config=replace(
                    base, model=model, latent_dim=latent_dim, seed=seed + fold_idx
                ),
                mask=train_mask,
            )
            prob = np.clip(
                predict_proba(fitted.params, factor_id, model=fitted.model),
                eps,
                1.0 - eps,
            )
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


def response_process_fit_diagnostics(
    responses: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray | None = None,
    item_type: str = "polytomous",
    response_process: str = "cumulative",
    eps: float = 1e-12,
    *,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
) -> FitDiagnostics:
    _validate_response_process(item_type, response_process)
    y, observed, prob = _prepare_categorical_response(
        responses, probabilities, mask, eps
    )
    _validate_category_count(item_type, prob.shape[2])
    onehot = np.eye(prob.shape[2], dtype=np.float64)[y]
    residual = (onehot - prob) * observed[:, :, None]
    pearson = np.where(observed[:, :, None], residual * residual / prob, 0.0)
    entry_chisq = pearson.sum(axis=2)
    log_prob = np.log(np.take_along_axis(prob, y[:, :, None], axis=2)[:, :, 0])
    entry_loglik = np.where(observed, log_prob, 0.0)

    itemfit = _categorical_axis_fit(observed, entry_loglik, entry_chisq, axis=0)
    personfit = _categorical_axis_fit(observed, entry_loglik, entry_chisq, axis=1)
    _attach_person_strata(
        personfit, group_id=group_id, cluster_id=cluster_id, n_persons=y.shape[0]
    )
    categoryfit = _category_fit(observed, onehot, prob, residual)
    loglik = float(entry_loglik.sum())
    n_observed = float(observed.sum())
    model_fit = {
        "loglik": loglik,
        "deviance": -2.0 * loglik,
        "n_observed": n_observed,
        "n_categories": float(prob.shape[2]),
        "pearson_chisq": float(entry_chisq.sum()),
        "mean_abs_category_residual": float(np.abs(residual[observed]).mean()),
    }
    return FitDiagnostics(
        itemfit=itemfit,
        personfit=personfit,
        model_fit=model_fit,
        categoryfit=categoryfit,
        groupfit=_categorical_stratum_fit(
            "group_id", group_id, observed, entry_loglik, entry_chisq
        ),
        clusterfit=_categorical_stratum_fit(
            "cluster_id", cluster_id, observed, entry_loglik, entry_chisq
        ),
        group_itemfit=_categorical_stratum_item_fit(
            "group_id", group_id, observed, entry_loglik, entry_chisq
        ),
        cluster_itemfit=_categorical_stratum_item_fit(
            "cluster_id", cluster_id, observed, entry_loglik, entry_chisq
        ),
    )


def response_process_dimensionality_diagnostics(
    responses: np.ndarray,
    candidate_probabilities: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    item_type: str = "polytomous",
    response_process: str = "cumulative",
    eps: float = 1e-12,
) -> DimensionalityDiagnostics:
    if not candidate_probabilities:
        raise ValueError("candidate_probabilities must not be empty")

    _validate_response_process(item_type, response_process)
    candidates: list[dict[str, float | str]] = []
    for idx, (label, probabilities) in enumerate(candidate_probabilities.items()):
        y, observed, prob = _prepare_categorical_response(
            responses, probabilities, mask, eps
        )
        _validate_category_count(item_type, prob.shape[2])
        entry_loglik, entry_chisq, residual = _categorical_entry_stats(
            y, observed, prob
        )
        n_observed = float(observed.sum())
        candidates.append(
            {
                "candidate_index": float(idx),
                "candidate_label": str(label),
                "heldout_loglik": float(entry_loglik.sum()),
                "heldout_deviance": float(-2.0 * entry_loglik.sum()),
                "n_observed": n_observed,
                "n_categories": float(prob.shape[2]),
                "pearson_chisq": float(entry_chisq.sum()),
                "mean_abs_category_residual": float(np.abs(residual[observed]).mean()),
            }
        )

    best = max(candidates, key=lambda row: float(row["heldout_loglik"]))
    return DimensionalityDiagnostics(candidates=candidates, best=best)


def fixed_item_calibration_diagnostics(
    responses: np.ndarray,
    candidate_probabilities: dict[str, np.ndarray],
    fixed_items: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    item_type: str = "polytomous",
    response_process: str = "cumulative",
    itemfit_penalty_weight: float = 1.0,
    eps: float = 1e-12,
) -> DimensionalityDiagnostics:
    """Select a candidate model using fixed-item likelihood and item-fit risk."""
    if not candidate_probabilities:
        raise ValueError("candidate_probabilities must not be empty")
    if itemfit_penalty_weight < 0:
        raise ValueError("itemfit_penalty_weight must be >= 0")

    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")

    fixed_idx = _fixed_item_indices(fixed_items, y.shape[1])
    fixed_responses = y[:, fixed_idx]
    if mask is None:
        fixed_mask = None
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != y.shape:
            raise ValueError("mask shape must match responses")
        fixed_mask = mask_arr[:, fixed_idx]
    observed = np.isfinite(fixed_responses) & (fixed_responses != -1)
    if fixed_mask is not None:
        observed &= fixed_mask
    if not np.any(observed):
        raise ValueError("fixed items contain no observed responses")

    candidates: list[dict[str, float | str]] = []
    for idx, (label, probabilities) in enumerate(candidate_probabilities.items()):
        candidate_label = str(label)
        if not candidate_label:
            raise ValueError("candidate label must not be empty")

        fixed_probabilities = _fixed_candidate_probabilities(
            probabilities, fixed_idx, y.shape
        )
        diagnostics = response_process_fit_diagnostics(
            fixed_responses,
            fixed_probabilities,
            mask=fixed_mask,
            item_type=item_type,
            response_process=response_process,
            eps=eps,
        )
        observed_item = diagnostics.itemfit["observed_count"] > 0
        outfit = diagnostics.itemfit["outfit_mnsq"][observed_item]
        kaefa_penalty = float(np.abs(outfit - 1.0).mean())
        loglik = float(diagnostics.model_fit["loglik"])
        calibration_score = loglik - float(itemfit_penalty_weight) * kaefa_penalty
        candidates.append(
            {
                "candidate_index": float(idx),
                "candidate_label": candidate_label,
                "fixed_item_count": float(fixed_idx.size),
                "fixed_item_observed_count": float(diagnostics.model_fit["n_observed"]),
                "heldout_loglik": loglik,
                "heldout_deviance": float(diagnostics.model_fit["deviance"]),
                "pearson_chisq": float(diagnostics.model_fit["pearson_chisq"]),
                "mean_abs_category_residual": float(
                    diagnostics.model_fit["mean_abs_category_residual"]
                ),
                "kaefa_itemfit_penalty": kaefa_penalty,
                "max_fixed_item_outfit_mnsq": float(outfit.max()),
                "itemfit_penalty_weight": float(itemfit_penalty_weight),
                "calibration_score": calibration_score,
            }
        )

    best = max(candidates, key=lambda row: float(row["calibration_score"]))
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
    denom = float(np.vdot(est_c, est_c))
    scale = float(np.sum(s) / denom) if denom > 1e-12 else 1.0
    aligned = scale * est_c @ rotation + true_mean
    return aligned[: len(true_xi)], aligned[len(true_xi) :]


def recovery_report(
    truth: MLSIRMParams, estimate: MLSIRMParams, align: bool = True
) -> RecoveryReport:
    est_xi = estimate.xi
    est_zeta = estimate.zeta
    if align:
        est_xi, est_zeta = align_latent_space(
            truth.xi, truth.zeta, estimate.xi, estimate.zeta
        )

    metrics = {
        "a_bias": _bias(truth.a, estimate.a),
        "a_rmse": _rmse(truth.a, estimate.a),
        "a_corr": _corr(truth.a, estimate.a),
        "b_bias": _bias(truth.b, estimate.b),
        "b_rmse": _rmse(truth.b, estimate.b),
        "b_corr": _corr(truth.b, estimate.b),
        "gamma_abs_error": float(abs(truth.gamma - estimate.gamma)),
        "gamma_relative_error": float(
            abs(truth.gamma - estimate.gamma) / max(abs(truth.gamma), 1e-12)
        ),
        "theta_rmse_standardized": _rmse(
            standardize(truth.theta), standardize(estimate.theta)
        ),
        "latent_coordinate_rmse": _rmse(
            np.vstack([truth.xi, truth.zeta]), np.vstack([est_xi, est_zeta])
        ),
        "person_item_distance_rmse": _distance_rmse(
            truth.xi, truth.zeta, estimate.xi, estimate.zeta
        ),
    }
    summary = {
        "parameter_rmse_mean": float(
            np.nanmean(
                [
                    metrics["a_rmse"],
                    metrics["b_rmse"],
                    metrics["theta_rmse_standardized"],
                ]
            )
        ),
        "latent_rmse": metrics["latent_coordinate_rmse"],
        "distance_rmse": metrics["person_item_distance_rmse"],
        "gamma_abs_error": metrics["gamma_abs_error"],
    }
    return RecoveryReport(summary=summary, metrics=metrics)


def _subset_params(
    params: MLSIRMParams, persons: np.ndarray | None, items: np.ndarray | None
) -> MLSIRMParams:
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
    safe_count = np.maximum(count, 1.0)
    safe_variance = np.maximum(variance_sum, 1e-12)
    return {
        "observed_count": count,
        "score": score,
        "expected_score": expected,
        "raw_residual": raw,
        "standardized_residual": raw / np.sqrt(safe_variance),
        "infit_mnsq": (residual * residual).sum(axis=axis) / safe_variance,
        "outfit_mnsq": pearson_sq.sum(axis=axis) / safe_count,
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

    unique_factors = np.unique(factors)
    # Optimized boolean mask matrix multiplication: Avoids slow python loops and intermediate
    # subset array allocations by converting aggregations to fast dense BLAS operations.
    mask = (factors[:, None] == unique_factors[None, :]).astype(np.float64)

    obs_sum = observed.sum(axis=0).astype(np.float64)
    y_obs_sum = (y * observed).sum(axis=0)
    prob_obs_sum = (prob * observed).sum(axis=0)
    res_sum = residual.sum(axis=0)
    var_obs_sum = (variance * observed).sum(axis=0)
    res_sq_sum = (residual * residual).sum(axis=0)
    pearson_sum = pearson_sq.sum(axis=0)

    count = obs_sum @ mask
    score = y_obs_sum @ mask
    expected_score = prob_obs_sum @ mask
    raw_residual = res_sum @ mask
    variance_sum = var_obs_sum @ mask
    infit_num = res_sq_sum @ mask
    outfit_num = pearson_sum @ mask

    safe_count = np.maximum(count, 1.0)
    safe_variance = np.maximum(variance_sum, 1e-12)
    return {
        "factor_id": unique_factors.astype(np.float64),
        "observed_count": count,
        "score": score,
        "expected_score": expected_score,
        "raw_residual": raw_residual,
        "standardized_residual": raw_residual / np.sqrt(safe_variance),
        "infit_mnsq": infit_num / safe_variance,
        "outfit_mnsq": outfit_num / safe_count,
    }


def _categorical_axis_fit(
    observed: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
    axis: int,
) -> dict[str, np.ndarray]:
    count = observed.sum(axis=axis).astype(np.float64)
    loglik = entry_loglik.sum(axis=axis)
    chisq = entry_chisq.sum(axis=axis)
    safe_count = np.maximum(count, 1.0)
    return {
        "observed_count": count,
        "loglik": loglik,
        "deviance": -2.0 * loglik,
        "pearson_chisq": chisq,
        "outfit_mnsq": chisq / safe_count,
    }


def _category_fit(
    observed: np.ndarray,
    onehot: np.ndarray,
    prob: np.ndarray,
    residual: np.ndarray,
) -> dict[str, np.ndarray]:
    observed_count = observed.sum(axis=0).astype(np.float64)
    obs_cast = observed.astype(prob.dtype, copy=False)
    score = np.einsum("ij,ijk->jk", obs_cast, onehot)
    expected = np.einsum("ij,ijk->jk", obs_cast, prob)
    variance = np.einsum("ij,ijk->jk", obs_cast, prob * (1.0 - prob))
    item_ids, category_ids = np.indices(score.shape)
    raw = residual.sum(axis=0)
    return {
        "item_id": item_ids.ravel().astype(np.float64),
        "category_id": category_ids.ravel().astype(np.float64),
        "observed_count": np.repeat(observed_count, score.shape[1]),
        "score": score.ravel(),
        "expected_score": expected.ravel(),
        "raw_residual": raw.ravel(),
        "standardized_residual": (raw / np.sqrt(np.maximum(variance, 1e-12))).ravel(),
    }


def _attach_person_strata(
    personfit: dict[str, np.ndarray],
    group_id: np.ndarray | None,
    cluster_id: np.ndarray | None,
    n_persons: int,
) -> None:
    if group_id is not None:
        personfit["group_id"] = _person_strata(group_id, n_persons, "group_id").astype(
            np.float64
        )
    if cluster_id is not None:
        personfit["cluster_id"] = _person_strata(
            cluster_id, n_persons, "cluster_id"
        ).astype(np.float64)


def _person_strata(values: np.ndarray, n_persons: int, name: str) -> np.ndarray:
    strata = np.asarray(values)
    if strata.shape != (n_persons,):
        raise ValueError(f"{name} length must match number of persons")
    return strata


def _binary_stratum_fit(
    id_name: str,
    strata: np.ndarray | None,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if strata is None:
        return None

    ids = _person_strata(strata, y.shape[0], id_name)
    loglik = np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0)
    rows = []
    for value in np.unique(ids):
        rows.append(
            _binary_scope_row(
                float(value),
                ids[:, None] == value,
                y,
                observed,
                prob,
                variance,
                residual,
                pearson_sq,
                loglik,
            )
        )
    return _binary_scope_table(id_name, rows)


def _binary_stratum_item_fit(
    id_name: str,
    strata: np.ndarray | None,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if strata is None:
        return None

    ids = _person_strata(strata, y.shape[0], id_name)
    loglik = np.where(observed, y * np.log(prob) + (1.0 - y) * np.log1p(-prob), 0.0)
    rows = []
    for value in np.unique(ids):
        row_mask = ids == value
        for item in range(y.shape[1]):
            scope = np.zeros_like(observed, dtype=bool)
            scope[row_mask, item] = True
            if np.any(observed & scope):
                rows.append(
                    (
                        float(value),
                        float(item),
                        *_binary_scope_row(
                            0.0,
                            scope,
                            y,
                            observed,
                            prob,
                            variance,
                            residual,
                            pearson_sq,
                            loglik,
                        )[1:],
                    )
                )
    return _binary_scope_item_table(id_name, rows)


def _binary_scope_row(
    id_value: float,
    scope: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
    variance: np.ndarray,
    residual: np.ndarray,
    pearson_sq: np.ndarray,
    loglik: np.ndarray,
) -> tuple[float, ...]:
    where = observed & scope
    count = float(where.sum())
    variance_sum = float((variance * where).sum())
    raw = float((residual * where).sum())
    chisq = float((pearson_sq * where).sum())
    ll = float((loglik * where).sum())
    return (
        id_value,
        count,
        float((y * where).sum()),
        float((prob * where).sum()),
        raw,
        raw / float(np.sqrt(max(variance_sum, 1e-12))),
        float((residual * residual * where).sum()) / max(variance_sum, 1e-12),
        chisq / max(count, 1.0),
        ll,
        -2.0 * ll,
        chisq,
    )


def _binary_scope_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "observed_count": table[:, 1],
        "score": table[:, 2],
        "expected_score": table[:, 3],
        "raw_residual": table[:, 4],
        "standardized_residual": table[:, 5],
        "infit_mnsq": table[:, 6],
        "outfit_mnsq": table[:, 7],
        "loglik": table[:, 8],
        "deviance": table[:, 9],
        "pearson_chisq": table[:, 10],
    }


def _binary_scope_item_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "item_id": table[:, 1],
        "observed_count": table[:, 2],
        "score": table[:, 3],
        "expected_score": table[:, 4],
        "raw_residual": table[:, 5],
        "standardized_residual": table[:, 6],
        "infit_mnsq": table[:, 7],
        "outfit_mnsq": table[:, 8],
        "loglik": table[:, 9],
        "deviance": table[:, 10],
        "pearson_chisq": table[:, 11],
    }


def _categorical_entry_stats(
    y: np.ndarray,
    observed: np.ndarray,
    prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    onehot = np.eye(prob.shape[2], dtype=np.float64)[y]
    residual = (onehot - prob) * observed[:, :, None]
    pearson = np.where(observed[:, :, None], residual * residual / prob, 0.0)
    entry_chisq = pearson.sum(axis=2)
    log_prob = np.log(np.take_along_axis(prob, y[:, :, None], axis=2)[:, :, 0])
    return np.where(observed, log_prob, 0.0), entry_chisq, residual


def _categorical_stratum_fit(
    id_name: str,
    strata: np.ndarray | None,
    observed: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if strata is None:
        return None

    ids = _person_strata(strata, observed.shape[0], id_name)
    rows = []
    for value in np.unique(ids):
        where = observed & (ids[:, None] == value)
        rows.append(
            _categorical_scope_row(float(value), where, entry_loglik, entry_chisq)
        )
    return _categorical_scope_table(id_name, rows)


def _categorical_stratum_item_fit(
    id_name: str,
    strata: np.ndarray | None,
    observed: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if strata is None:
        return None

    ids = _person_strata(strata, observed.shape[0], id_name)
    rows = []
    for value in np.unique(ids):
        row_mask = ids == value
        for item in range(observed.shape[1]):
            where = np.zeros_like(observed, dtype=bool)
            where[row_mask, item] = observed[row_mask, item]
            if np.any(where):
                rows.append(
                    (
                        float(value),
                        float(item),
                        *_categorical_scope_row(0.0, where, entry_loglik, entry_chisq)[
                            1:
                        ],
                    )
                )
    return _categorical_scope_item_table(id_name, rows)


def _categorical_scope_row(
    id_value: float,
    where: np.ndarray,
    entry_loglik: np.ndarray,
    entry_chisq: np.ndarray,
) -> tuple[float, ...]:
    count = float(where.sum())
    loglik = float(entry_loglik[where].sum())
    chisq = float(entry_chisq[where].sum())
    return (id_value, count, loglik, -2.0 * loglik, chisq, chisq / max(count, 1.0))


def _categorical_scope_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "observed_count": table[:, 1],
        "loglik": table[:, 2],
        "deviance": table[:, 3],
        "pearson_chisq": table[:, 4],
        "outfit_mnsq": table[:, 5],
    }


def _categorical_scope_item_table(
    id_name: str, rows: list[tuple[float, ...]]
) -> dict[str, np.ndarray]:
    table = np.asarray(rows, dtype=np.float64)
    return {
        id_name: table[:, 0],
        "item_id": table[:, 1],
        "observed_count": table[:, 2],
        "loglik": table[:, 3],
        "deviance": table[:, 4],
        "pearson_chisq": table[:, 5],
        "outfit_mnsq": table[:, 6],
    }


def _fixed_item_indices(fixed_items: np.ndarray | None, n_items: int) -> np.ndarray:
    if fixed_items is None:
        return np.arange(n_items, dtype=np.int64)

    values = np.asarray(fixed_items)
    if values.ndim != 1:
        raise ValueError("fixed_items must be a 1D boolean mask or item-index vector")
    if values.dtype == np.bool_:
        if values.shape[0] != n_items:
            raise ValueError(
                "fixed_items boolean mask length must match number of items"
            )
        indices = np.flatnonzero(values)
    else:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("fixed_items index vector must contain integers")
        indices = values.astype(np.int64, copy=False)

    if indices.size == 0:
        raise ValueError("fixed_items must select at least one item")
    if np.any((indices < 0) | (indices >= n_items)):
        raise ValueError("fixed_items index vector contains an out-of-range item")
    if np.unique(indices).size != indices.size:
        raise ValueError("fixed_items index vector must not contain duplicates")
    return indices


def _fixed_candidate_probabilities(
    probabilities: np.ndarray, fixed_idx: np.ndarray, response_shape: tuple[int, int]
) -> np.ndarray:
    prob = np.asarray(probabilities)
    if prob.ndim not in {2, 3}:
        raise ValueError(
            "candidate probabilities must have shape persons x items or persons x items x categories"
        )
    if prob.shape[:2] != response_shape:
        raise ValueError("candidate probabilities shape must match responses")
    if prob.ndim == 2:
        return prob[:, fixed_idx]
    return prob[:, fixed_idx, :]


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


def _validation_folds(
    observed: np.ndarray, k_folds: int, seed: int
) -> list[np.ndarray]:
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2")

    row_counts = observed.sum(axis=1)
    col_counts = observed.sum(axis=0)
    eligible = np.argwhere(
        observed & (row_counts[:, None] > 1) & (col_counts[None, :] > 1)
    )
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
        # ~train.any() is faster than train.sum() == 0 as it avoids allocating an integer array
        mask[~train.any(axis=1), :] = False
        mask[:, ~train.any(axis=0)] = False
        if not np.any(mask):
            raise ValueError("fold validation set is empty; reduce k_folds")
        folds.append(mask)
    return folds


def _accumulate_heldout(
    totals: dict[str, float], y: np.ndarray, mask: np.ndarray, prob: np.ndarray
) -> None:
    yy = y[mask]
    pp = prob[mask]
    residual = yy - pp
    totals["loglik"] += float((yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp)).sum())
    totals["abs_residual"] += float(np.abs(residual).sum())
    totals["sq_residual"] += float((residual * residual).sum())
    totals["n"] += float(mask.sum())


def _validate_response_process(item_type: str, response_process: str) -> None:
    if item_type not in {"dichotomous", "polytomous"}:
        raise ValueError("item_type must be dichotomous or polytomous")
    if response_process not in {"ideal_point", "cumulative"}:
        raise ValueError("response_process must be ideal_point or cumulative")


def _validate_category_count(item_type: str, n_categories: int) -> None:
    if item_type == "dichotomous" and n_categories != 2:
        raise ValueError("dichotomous diagnostics require exactly 2 categories")
    if item_type == "polytomous" and n_categories < 3:
        raise ValueError("polytomous diagnostics require at least 3 categories")


def _prepare_categorical_response(
    responses: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")

    prob = np.asarray(probabilities, dtype=np.float64)
    if prob.ndim == 2:
        if prob.shape != y.shape:
            raise ValueError("probabilities shape must match responses")
        prob = np.stack([1.0 - prob, prob], axis=2)
    if prob.ndim != 3 or prob.shape[:2] != y.shape:
        raise ValueError("probabilities must have shape persons x items x categories")

    observed = (
        np.isfinite(y) & (y != -1) if mask is None else np.asarray(mask, dtype=bool)
    )
    if observed.shape != y.shape:
        raise ValueError("mask shape must match responses")
    observed &= np.isfinite(y) & (y != -1)
    if not np.any(observed):
        raise ValueError("responses contain no observed entries")

    yy = np.where(observed, y, 0).astype(np.int64)
    if np.any(observed & ((yy < 0) | (yy >= prob.shape[2]))):
        raise ValueError("observed responses must be valid category ids")

    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum(axis=2, keepdims=True)
    return yy, observed, prob


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


def _distance_rmse(
    true_xi: np.ndarray, true_zeta: np.ndarray, est_xi: np.ndarray, est_zeta: np.ndarray
) -> float:
    # Optimized distance calculation: replace memory allocation in (x * x).sum(axis=1) with np.einsum
    true_sq_xi = np.einsum("ij,ij->i", true_xi, true_xi)[:, None]
    true_sq_zeta = np.einsum("ij,ij->i", true_zeta, true_zeta)[None, :]
    true_d = np.sqrt(
        np.maximum(true_sq_xi - 2 * np.dot(true_xi, true_zeta.T) + true_sq_zeta, 0.0)
    )

    est_sq_xi = np.einsum("ij,ij->i", est_xi, est_xi)[:, None]
    est_sq_zeta = np.einsum("ij,ij->i", est_zeta, est_zeta)[None, :]
    est_d = np.sqrt(
        np.maximum(est_sq_xi - 2 * np.dot(est_xi, est_zeta.T) + est_sq_zeta, 0.0)
    )

    return _rmse(true_d, est_d)
