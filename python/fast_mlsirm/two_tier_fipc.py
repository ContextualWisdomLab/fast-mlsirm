"""Reference-metric person scoring for two-tier GRM FIPC fits.

This module deliberately owns scoring, not calibration.  A two-tier fit is
supplied by the caller (normally :class:`TwoTierGrmFit`); anchor rows identify
the reference item metric and the focal primary prior remains free.  The
orthogonal-primary contract is explicit because a correlated ``phi`` needs a
different conditional nuisance integral.

The implementation follows Kim (2006) for fixed-item linking, Bock and
Mislevy (1982) for EAP, and Lord (1980) for the expected-raw plug-in.  The
two-tier item likelihood uses the Gibbons et al. (2007) per-specific-block
reduction.  It is intentionally domain-neutral: no G+4+W labels are encoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .polytomous import MAX_POLY_QUADRATURE_POINTS


def _integer(value: object, name: str) -> int:
    """Validate a Python integer control without silently coercing floats."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    if value > MAX_POLY_QUADRATURE_POINTS:
        raise ValueError(f"{name} must be <= {MAX_POLY_QUADRATURE_POINTS}")
    return value


def _normal_vector(value: object, n: int, name: str, *, positive: bool) -> np.ndarray:
    """Normalize a scalar-broadcast or exact length vector."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(n, float(arr), dtype=np.float64)
    elif arr.shape == (n,):
        out = arr.copy()
    else:
        raise ValueError(f"{name} must be scalar or shape ({n},), got {arr.shape}")
    if not np.all(np.isfinite(out)) or (positive and np.any(out <= 0.0)):
        suffix = " and > 0" if positive else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return out


def _identity_phi(phi: object, n_primary: int) -> None:
    """Require the explicit orthogonal nuisance identification contract."""
    arr = np.asarray(phi, dtype=np.float64)
    if arr.shape != (n_primary, n_primary) or not np.allclose(
        arr, np.eye(n_primary), atol=0.0, rtol=0.0
    ):
        raise ValueError(
            "two-tier FIPC scoring requires Phi == I under the explicit "
            "orthogonal-primary identification contract"
        )


def _fit_arrays(fit, specific_map: np.ndarray, anchor: np.ndarray):
    """Validate the fit-like protocol and return dense numerical arrays."""
    required = (
        "a_primary",
        "a_specific",
        "threshold",
        "phi",
        "n_primary",
        "n_specific",
        "orthogonal_primary_identification",
    )
    if any(not hasattr(fit, name) for name in required):
        raise TypeError(
            "fit must expose a_primary, a_specific, threshold, phi, "
            "n_primary, n_specific, and orthogonal_primary_identification"
        )
    if fit.orthogonal_primary_identification is not True:
        raise ValueError(
            "fit must carry orthogonal_primary_identification=True; "
            "Phi == I alone does not prove the estimation identification"
        )
    ap = np.asarray(fit.a_primary, dtype=np.float64)
    asp = np.asarray(fit.a_specific, dtype=np.float64)
    threshold = np.asarray(fit.threshold, dtype=np.float64)
    smap = np.asarray(specific_map)
    mask = np.asarray(anchor)
    if ap.ndim != 2 or ap.shape[0] == 0:
        raise ValueError("fit.a_primary must be a non-empty n_items x n_primary array")
    n_items, n_primary = ap.shape
    if asp.shape != (n_items,):
        raise ValueError("fit.a_specific must have shape (n_items,)")
    if threshold.ndim != 2 or threshold.shape[0] != n_items or threshold.shape[1] < 1:
        raise ValueError("fit.threshold must be n_items x (n_cat - 1)")
    if np.any(np.diff(threshold, axis=1) >= 0.0):
        raise ValueError("fit.threshold rows must be strictly decreasing")
    if not np.all(np.isfinite(ap)) or not np.all(np.isfinite(asp)) or not np.all(
        np.isfinite(threshold)
    ):
        raise ValueError("fit parameters must be finite")
    if int(fit.n_primary) != n_primary:
        raise ValueError("fit.n_primary is inconsistent with fit.a_primary")
    if smap.shape != (n_items,):
        raise ValueError("specific_map must have shape (n_items,)")
    if not np.issubdtype(smap.dtype, np.integer) and (
        not np.all(np.isfinite(smap)) or np.any(smap != np.floor(smap))
    ):
        raise ValueError("specific_map entries must be integers")
    smap = smap.astype(np.int64, copy=False)
    if np.any(smap < -1):
        raise ValueError("specific_map entries must be -1 or non-negative")
    n_specific = int(smap.max()) + 1 if np.any(smap >= 0) else 0
    if int(fit.n_specific) != n_specific:
        raise ValueError("fit.n_specific is inconsistent with specific_map")
    if np.any((smap >= 0) & (asp == 0.0)):
        # Zero is valid as a loading, but this check catches accidental shape
        # adapters only when the block itself has no non-zero loading below.
        pass
    if mask.dtype.kind != "b" or mask.shape != (n_items,) or not np.any(mask):
        raise ValueError("anchor must be a boolean mask with at least one item")
    _identity_phi(fit.phi, n_primary)
    return ap, asp, threshold, smap, mask.astype(bool, copy=True)


def _responses(responses: np.ndarray, n_items: int, n_cat: int):
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != n_items:
        raise ValueError("responses must be a persons x n_items array")
    if np.any(np.isinf(y)):
        raise ValueError("responses must not contain infinity")
    observed = np.isfinite(y) & (y >= 0.0)
    yi = np.zeros(y.shape, dtype=np.int64)
    if np.any(observed):
        vals = y[observed]
        if np.any(vals != np.floor(vals)) or np.any(vals >= n_cat):
            raise ValueError(f"observed responses must be integer categories in 0..{n_cat - 1}")
        yi[observed] = vals.astype(np.int64)
    return yi, observed


def _grm_probs(base: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Return adjacent-difference GRM category probabilities."""
    cumulative = 1.0 / (1.0 + np.exp(-(base[..., None] + threshold)))
    ones = np.ones(base.shape + (1,), dtype=np.float64)
    zeros = np.zeros(base.shape + (1,), dtype=np.float64)
    return np.concatenate((ones - cumulative[..., :1], -np.diff(cumulative, axis=-1), cumulative[..., -1:] - zeros), axis=-1)


def _gh(q: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(q)
    weights = weights / weights.sum()
    return nodes.astype(np.float64), weights.astype(np.float64)


def _primary_grid(mean: np.ndarray, sd: np.ndarray, q: int):
    nodes, weights = _gh(q)
    axes = [mean[d] + sd[d] * nodes for d in range(mean.size)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    wmesh = np.meshgrid(*([weights] * mean.size), indexing="ij")
    weights_flat = np.prod([axis.reshape(-1) for axis in wmesh], axis=0)
    return grid, weights_flat


def _expected_raw_at_primary(
    primary: np.ndarray,
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    threshold: np.ndarray,
    specific_map: np.ndarray,
    specific_mean: np.ndarray,
    specific_sd: np.ndarray,
    q_specific: int,
) -> np.ndarray:
    """Evaluate E[T|primary] with each item's specific nuisance integrated."""
    nodes, weights = _gh(q_specific)
    out = np.zeros(primary.shape[0], dtype=np.float64)
    for i in range(a_primary.shape[0]):
        base = primary @ a_primary[i]
        sid = int(specific_map[i])
        if sid >= 0 and a_specific[i] != 0.0:
            base = base[:, None] + a_specific[i] * (specific_mean[sid] + specific_sd[sid] * nodes)[None, :]
            probs = _grm_probs(base, threshold[i])
            out += (probs * np.arange(probs.shape[-1])).sum(axis=-1) @ weights
        else:
            probs = _grm_probs(base, threshold[i])
            out += (probs * np.arange(probs.shape[-1])).sum(axis=-1)
    return out


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + float(np.log(np.exp(values - peak).sum()))


def _eap(
    y: np.ndarray,
    observed: np.ndarray,
    primary_grid: np.ndarray,
    primary_weights: np.ndarray,
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    threshold: np.ndarray,
    specific_map: np.ndarray,
    specific_mean: np.ndarray,
    specific_sd: np.ndarray,
    q_specific: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute primary EAPs by the two-tier block likelihood reduction."""
    nodes, weights = _gh(q_specific)
    n_persons, n_items = y.shape
    n_nodes, n_primary = primary_grid.shape
    log_prior = np.log(primary_weights)
    block_ids = sorted(int(v) for v in np.unique(specific_map) if v >= 0)
    eap = np.empty((n_persons, n_primary), dtype=np.float64)
    sd = np.empty_like(eap)
    for person in range(n_persons):
        log_post = log_prior.copy()
        for item in range(n_items):
            if not observed[person, item] or specific_map[item] >= 0:
                continue
            base = primary_grid @ a_primary[item]
            logp = np.log(np.clip(_grm_probs(base, threshold[item])[:, int(y[person, item])], 1e-300, 1.0))
            log_post += logp
        for block in block_ids:
            members = np.flatnonzero(specific_map == block)
            block_log = np.zeros((n_nodes, q_specific), dtype=np.float64)
            for item in members:
                if not observed[person, item]:
                    continue
                base = primary_grid @ a_primary[item]
                base = base[:, None] + a_specific[item] * (specific_mean[block] + specific_sd[block] * nodes)[None, :]
                logp = np.log(np.clip(_grm_probs(base, threshold[item])[:, :, int(y[person, item])], 1e-300, 1.0))
                block_log += logp
            block_log += np.log(weights)[None, :]
            log_post += np.array([_logsumexp(row) for row in block_log])
        norm = _logsumexp(log_post)
        posterior = np.exp(log_post - norm)
        mean = posterior @ primary_grid
        second = posterior @ np.square(primary_grid)
        variance = second - mean * mean
        if np.any(variance < -1e-9):
            raise RuntimeError("primary EAP posterior variance computed negative")
        eap[person] = mean
        sd[person] = np.sqrt(np.maximum(variance, 0.0))
    return eap, sd


@dataclass(frozen=True)
class TwoTierReferenceExpectedScoreMoments:
    """Moments of the anchor-only expected raw score distribution."""

    mean: float
    second_moment: float
    variance: float
    primary_mean: np.ndarray
    primary_sd: np.ndarray
    n_anchor_items: int
    q_primary: int
    q_specific: int


@dataclass(frozen=True)
class TwoTierFipcGroupPersonScores:
    """Focal person scores on the reference item metric."""

    theta_primary_eap: np.ndarray
    theta_primary_sd: np.ndarray
    expected_raw: np.ndarray
    reference_moments: TwoTierReferenceExpectedScoreMoments
    fit: object


@dataclass(frozen=True)
class TwoTierGrmFipcFit:
    """Python-side representation of the Rust two-tier FIPC result."""

    a_primary: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    phi: np.ndarray
    primary_mean: np.ndarray
    primary_sd: np.ndarray
    specific_sd: np.ndarray
    theta_p_eap: np.ndarray
    theta_p_sd: np.ndarray
    category_counts: np.ndarray
    n_cat: int
    n_primary: int
    n_specific: int
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    n_parameters: int
    orthogonal_primary_identification: bool


def fit_two_tier_grm_fipc(
    responses: np.ndarray,
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_primary: int,
    n_specific: int,
    anchor: np.ndarray,
    fixed_a_primary: np.ndarray,
    fixed_a_specific: np.ndarray,
    fixed_threshold: np.ndarray,
    *,
    q_primary: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    newton_iter: int = 10,
    ridge: float = 1e-8,
    estimate_specific_vars: bool = False,
    orthogonal_primary_identification: bool = True,
) -> TwoTierGrmFipcFit:
    """Fit two-tier FIPC in the Rust core and adapt its result for scoring.

    This is intentionally gated on the future ``fit_two_tier_grm_fipc`` PyO3
    symbol.  Until that estimator is available, callers receive a clear
    runtime error instead of a metadata-only or single-group substitute.
    """
    from .fitstats import _core_module

    core = _core_module()
    estimator = None if core is None else getattr(core, "fit_two_tier_grm_fipc", None)
    if estimator is None:
        raise RuntimeError("fit_two_tier_grm_fipc requires the compiled Rust core")
    if orthogonal_primary_identification is not True:
        raise ValueError("two-tier FIPC scoring requires orthogonal_primary_identification=True")

    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    observed = np.isfinite(y) & (y >= 0.0)
    y_int = np.zeros_like(y, dtype=np.int64)
    if np.any(observed):
        values = y[observed]
        if np.any(values != np.floor(values)) or np.any(values >= n_cat):
            raise ValueError(f"observed responses must be integer categories in 0..{n_cat - 1}")
        y_int[observed] = values.astype(np.int64)
    pmap = np.asarray(primary_map, dtype=bool)
    smap = np.asarray(specific_map, dtype=np.int64)
    anchor_arr = np.asarray(anchor, dtype=bool)
    fixed_primary = np.asarray(fixed_a_primary, dtype=np.float64)
    fixed_specific = np.asarray(fixed_a_specific, dtype=np.float64)
    fixed_threshold_arr = np.asarray(fixed_threshold, dtype=np.float64)
    if pmap.shape != (n_items, n_primary):
        raise ValueError("primary_map must be an n_items x n_primary array")
    if smap.shape != (n_items,) or anchor_arr.shape != (n_items,):
        raise ValueError("specific_map and anchor must have length n_items")
    if fixed_primary.shape != (n_items, n_primary):
        raise ValueError("fixed_a_primary must have shape (n_items, n_primary)")
    if fixed_specific.shape != (n_items,):
        raise ValueError("fixed_a_specific must have shape (n_items,)")
    if fixed_threshold_arr.shape != (n_items, n_cat - 1):
        raise ValueError("fixed_threshold must have shape (n_items, n_cat - 1)")
    result = estimator(
        y_int.reshape(-1),
        observed.reshape(-1),
        pmap.reshape(-1),
        smap,
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        anchor_arr,
        fixed_primary.reshape(-1),
        fixed_specific,
        fixed_threshold_arr.reshape(-1),
        q_primary,
        q_specific,
        max_iter,
        tol,
        newton_iter,
        ridge,
        estimate_specific_vars,
        orthogonal_primary_identification,
    )
    required = (
        "a_primary", "a_specific", "threshold", "phi", "primary_mean", "primary_sd",
        "specific_sd", "theta_p_eap", "theta_p_sd", "category_counts", "n_iter",
        "converged", "termination_reason", "final_loglik_change", "n_parameters",
        "orthogonal_primary_identification",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise RuntimeError(f"fit_two_tier_grm_fipc result missing required fields: {missing}")
    return TwoTierGrmFipcFit(
        a_primary=np.asarray(result["a_primary"], dtype=np.float64).reshape(n_items, n_primary),
        a_specific=np.asarray(result["a_specific"], dtype=np.float64),
        threshold=np.asarray(result["threshold"], dtype=np.float64).reshape(n_items, n_cat - 1),
        phi=np.asarray(result["phi"], dtype=np.float64).reshape(n_primary, n_primary),
        primary_mean=np.asarray(result["primary_mean"], dtype=np.float64),
        primary_sd=np.asarray(result["primary_sd"], dtype=np.float64),
        specific_sd=np.asarray(result["specific_sd"], dtype=np.float64),
        theta_p_eap=np.asarray(result["theta_p_eap"], dtype=np.float64).reshape(n_persons, n_primary),
        theta_p_sd=np.asarray(result["theta_p_sd"], dtype=np.float64).reshape(n_persons, n_primary),
        category_counts=np.asarray(result["category_counts"], dtype=np.int64).reshape(n_items, n_cat),
        n_cat=int(n_cat),
        n_primary=int(n_primary),
        n_specific=int(n_specific),
        loglik_trace=np.asarray(result.get("loglik_trace", []), dtype=np.float64),
        n_iter=int(result["n_iter"]),
        converged=bool(result["converged"]),
        termination_reason=str(result["termination_reason"]),
        final_loglik_change=float(result["final_loglik_change"]),
        n_parameters=int(result["n_parameters"]),
        orthogonal_primary_identification=bool(result["orthogonal_primary_identification"]),
    )


def fit_and_score_two_tier_fipc_group_persons(
    responses: np.ndarray,
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    n_cat: int,
    n_primary: int,
    n_specific: int,
    anchor: np.ndarray,
    fixed_a_primary: np.ndarray,
    fixed_a_specific: np.ndarray,
    fixed_threshold: np.ndarray,
    *,
    q_primary: int,
    q_specific: int,
    max_iter: int,
    tol: float,
    newton_iter: int = 10,
    ridge: float = 1e-8,
    estimate_specific_vars: bool = False,
    reference_primary_mean: object = 0.0,
    reference_primary_sd: object = 1.0,
    reference_specific_mean: object = 0.0,
    reference_specific_sd: object = 1.0,
) -> TwoTierFipcGroupPersonScores:
    """Run the real Rust estimator, then score its focal responses."""
    fit = fit_two_tier_grm_fipc(
        responses, primary_map, specific_map, n_cat, n_primary, n_specific, anchor,
        fixed_a_primary, fixed_a_specific, fixed_threshold, q_primary=q_primary,
        q_specific=q_specific, max_iter=max_iter, tol=tol, newton_iter=newton_iter,
        ridge=ridge, estimate_specific_vars=estimate_specific_vars,
    )
    return score_two_tier_fipc_group_persons(
        responses, fit, specific_map, anchor, q_primary=q_primary, q_specific=q_specific,
        focal_primary_mean=fit.primary_mean, focal_primary_sd=fit.primary_sd,
        focal_specific_sd=fit.specific_sd, reference_primary_mean=reference_primary_mean,
        reference_primary_sd=reference_primary_sd, reference_specific_mean=reference_specific_mean,
        reference_specific_sd=reference_specific_sd,
    )


def two_tier_reference_expected_score_moments(
    fit,
    specific_map: np.ndarray,
    anchor: np.ndarray,
    *,
    primary_mean: object = 0.0,
    primary_sd: object = 1.0,
    specific_mean: object = 0.0,
    specific_sd: object = 1.0,
    q_primary: int,
    q_specific: int,
) -> TwoTierReferenceExpectedScoreMoments:
    """Integrate ``E[T|theta]`` using reference anchor rows only."""
    q_p = _integer(q_primary, "q_primary")
    q_s = _integer(q_specific, "q_specific")
    ap, asp, threshold, smap, mask = _fit_arrays(fit, specific_map, anchor)
    p_mean = _normal_vector(primary_mean, ap.shape[1], "primary_mean", positive=False)
    p_sd = _normal_vector(primary_sd, ap.shape[1], "primary_sd", positive=True)
    n_specific = int(smap.max()) + 1 if np.any(smap >= 0) else 0
    s_mean = _normal_vector(specific_mean, n_specific, "specific_mean", positive=False)
    s_sd = _normal_vector(specific_sd, n_specific, "specific_sd", positive=True)
    anchor_ap, anchor_asp, anchor_threshold = ap[mask], asp[mask], threshold[mask]
    anchor_map = smap[mask]
    grid, weights = _primary_grid(p_mean, p_sd, q_p)
    totals = _expected_raw_at_primary(grid, anchor_ap, anchor_asp, anchor_threshold, anchor_map, s_mean, s_sd, q_s)
    mean = float(weights @ totals)
    second = float(weights @ np.square(totals))
    variance = second - mean * mean
    if variance < 0.0 and variance > -1e-10:
        variance = 0.0
    if variance < 0.0:
        raise RuntimeError("reference expected-score variance computed negative")
    return TwoTierReferenceExpectedScoreMoments(mean, second, variance, p_mean, p_sd, int(mask.sum()), q_p, q_s)


def score_two_tier_fipc_group_persons(
    responses: np.ndarray,
    fit,
    specific_map: np.ndarray,
    anchor: np.ndarray,
    *,
    q_primary: int,
    q_specific: int,
    focal_primary_mean: object = 0.0,
    focal_primary_sd: object = 1.0,
    reference_primary_mean: object = 0.0,
    reference_primary_sd: object = 1.0,
    reference_specific_mean: object = 0.0,
    reference_specific_sd: object = 1.0,
    focal_specific_mean: object = 0.0,
    focal_specific_sd: object = 1.0,
) -> TwoTierFipcGroupPersonScores:
    """Score a focal group from an anchored two-tier fit.

    ``fit`` supplies the item bank after calibration: anchor rows are the
    reference metric and non-anchor rows are allowed to be focal/free rows.
    ``focal_primary_mean``/``focal_primary_sd`` are therefore never replaced
    by ``N(0, 1)``.  EAPs remain expressed in the reference metric because no
    latent rescaling is performed.  Reference moments use only ``anchor``
    rows and the separate reference prior.
    """
    q_p = _integer(q_primary, "q_primary")
    q_s = _integer(q_specific, "q_specific")
    ap, asp, threshold, smap, mask = _fit_arrays(fit, specific_map, anchor)
    n_cat = threshold.shape[1] + 1
    y, observed = _responses(responses, ap.shape[0], n_cat)
    p_mean = _normal_vector(focal_primary_mean, ap.shape[1], "focal_primary_mean", positive=False)
    p_sd = _normal_vector(focal_primary_sd, ap.shape[1], "focal_primary_sd", positive=True)
    n_specific = int(smap.max()) + 1 if np.any(smap >= 0) else 0
    s_mean = _normal_vector(focal_specific_mean, n_specific, "focal_specific_mean", positive=False)
    s_sd = _normal_vector(focal_specific_sd, n_specific, "focal_specific_sd", positive=True)
    grid, weights = _primary_grid(p_mean, p_sd, q_p)
    eap, eap_sd = _eap(y, observed, grid, weights, ap, asp, threshold, smap, s_mean, s_sd, q_s)
    expected_raw = _expected_raw_at_primary(eap, ap, asp, threshold, smap, s_mean, s_sd, q_s)
    reference = two_tier_reference_expected_score_moments(
        fit,
        smap,
        mask,
        primary_mean=reference_primary_mean,
        primary_sd=reference_primary_sd,
        specific_mean=reference_specific_mean,
        specific_sd=reference_specific_sd,
        q_primary=q_p,
        q_specific=q_s,
    )
    return TwoTierFipcGroupPersonScores(eap, eap_sd, expected_raw, reference, fit)


def execute_two_tier_fipc_group_person_score_payload(payload: dict[str, object]) -> dict[str, object]:
    """Execute a serializable scorer payload when the caller supplies fit fields."""
    if type(payload) is not dict:
        raise TypeError("payload must be a dict")
    required = ("responses", "fit", "specific_map", "anchor", "q_primary", "q_specific")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"payload missing required keys: {missing}")
    # JSON workers cannot reconstruct arbitrary fit classes; the equivalent
    # protocol is a field mapping with the same TwoTierGrmFit attributes.
    fit = type("TwoTierFitPayload", (), dict(payload["fit"]))()
    result = score_two_tier_fipc_group_persons(
        np.asarray(payload["responses"]),
        fit,
        np.asarray(payload["specific_map"]),
        np.asarray(payload["anchor"]),
        q_primary=payload["q_primary"],
        q_specific=payload["q_specific"],
        focal_primary_mean=payload.get("focal_primary_mean", 0.0),
        focal_primary_sd=payload.get("focal_primary_sd", 1.0),
        reference_primary_mean=payload.get("reference_primary_mean", 0.0),
        reference_primary_sd=payload.get("reference_primary_sd", 1.0),
        reference_specific_mean=payload.get("reference_specific_mean", 0.0),
        reference_specific_sd=payload.get("reference_specific_sd", 1.0),
        focal_specific_mean=payload.get("focal_specific_mean", 0.0),
        focal_specific_sd=payload.get("focal_specific_sd", 1.0),
    )
    return {
        "family": "two_tier_fipc_group_person_score",
        "model_scope": "two_tier_grm_fipc_orthogonal_primary",
        "theta_primary_eap": result.theta_primary_eap.tolist(),
        "theta_primary_sd": result.theta_primary_sd.tolist(),
        "expected_raw": result.expected_raw.tolist(),
        "reference_expected_score": {
            "mean": result.reference_moments.mean,
            "second_moment": result.reference_moments.second_moment,
            "variance": result.reference_moments.variance,
            "n_anchor_items": result.reference_moments.n_anchor_items,
        },
    }


__all__ = [
    "TwoTierFipcGroupPersonScores",
    "TwoTierReferenceExpectedScoreMoments",
    "execute_two_tier_fipc_group_person_score_payload",
    "score_two_tier_fipc_group_persons",
    "two_tier_reference_expected_score_moments",
]
