"""All-fixed-item orthogonal two-tier focal fitting and person scoring.

Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. Psychometrika, 75, 581-612. doi:10.1007/s11336-010-9178-0.
Opened p. 587 equations 4-6 and pp. 608-609 Appendices A/B define Gaussian
latent distributions, latent-density M-steps and posterior moments. Independent
primaries are an explicit restriction. The Gaussian EM update and affine GH
substitution are derived in the Rust docstrings; this is not Kim's fixed-node
weight update or empirical proof of focal calibration validity. The caller
retains immutable item bank/maps, input hashes and population-fit provenance.
"""

from dataclasses import dataclass

import numpy as np

from .fitstats import _core_module
from .polytomous import _bounded_integer, _positive_real


@dataclass
class TwoTierGrmPersonScores:
    """Native posterior moments, persons x (primaries followed by specifics).

    Cai (2010), p. 609 Appendix B. SD conditions on the supplied item bank and
    distribution; it excludes their estimation uncertainty. Second is E[t²|Y].
    """

    mean: np.ndarray
    second: np.ndarray
    sd: np.ndarray
    loglik: float
    backend: str = "cpu"
    gpu_memory_budget_bytes: int | None = None


@dataclass
class TwoTierGrmFocalFit:
    """Native Gaussian focal fit and actual termination/settings record.

    Cai (2010), pp. 608-609 Appendix A supplies the Gaussian M-step objective.
    A nonconverged fit retains diagnostic scores; it is not an accepted model.
    """

    latent_mean: np.ndarray
    latent_sd: np.ndarray
    scores: TwoTierGrmPersonScores
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    initial_mean: np.ndarray
    initial_sd: np.ndarray
    q_primary: int
    q_specific: int
    max_iter: int
    tol: float


def _real_array(values, name, shape):
    """Validate real Gaussian/item arrays from Cai (2010), p. 587 eqs. 5-6.

    NumPy conversion is glue; finiteness/shape are checked before dispatch.
    """
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf" or raw.shape != shape:
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _native_inputs(
    responses,
    primary_map,
    specific_map,
    a_primary,
    a_specific,
    threshold,
    latent_mean,
    latent_sd,
    *,
    n_cat,
    n_primary,
    n_specific,
    q_primary,
    q_specific,
):
    """Shape the declared two-tier model (Cai, 2010, p. 587, eqs. 4-7).

    Maps/categories are checked before narrowing casts. Native integer-width
    bounds are transport constraints, not scientific settings or node caps.
    NaN/-1 mark missing; invalid categories/complex/object arrays are rejected.
    """
    usize = int(np.iinfo(np.uintp).max)
    n_cat = _bounded_integer(n_cat, "n_cat", 2, usize)
    p = _bounded_integer(n_primary, "n_primary", 1, usize)
    s = _bounded_integer(n_specific, "n_specific", 1, usize)
    qp = _bounded_integer(q_primary, "q_primary", 1, usize)
    qs = _bounded_integer(q_specific, "q_specific", 1, usize)
    raw = np.asarray(responses)
    if raw.dtype.kind not in "biuf" or raw.ndim != 2 or 0 in raw.shape:
        raise ValueError("responses must be a nonempty real persons x items array")
    missing = (raw == -1) | np.isnan(raw)
    observed = ~missing
    vals = raw[observed]
    too_wide = (
        (vals >= 2**63).any()
        if vals.dtype.kind == "f"
        else (vals > np.iinfo(np.int64).max).any()
    )
    fractional = vals.dtype.kind == "f" and (vals != np.floor(vals)).any()
    if vals.size and (
        not np.isfinite(vals).all()
        or (vals < 0).any()
        or (vals >= n_cat).any()
        or too_wide
        or fractional
    ):
        raise ValueError(
            "observed responses must be integer categories in 0..n_cat-1 fitting int64"
        )
    y = np.ascontiguousarray(np.where(observed, raw, 0), dtype=np.int64)
    n, items = raw.shape
    pm = np.asarray(primary_map)
    if pm.dtype != np.dtype(bool) or pm.shape != (items, p):
        raise ValueError("primary_map must be an items x n_primary boolean array")
    sm = np.asarray(specific_map)
    if sm.dtype.kind not in "iu" or sm.shape != (items,):
        raise ValueError("specific_map must be an integer array of length n_items")
    if (sm < -1).any() or (sm >= s).any() or (sm > np.iinfo(np.int32).max).any():
        raise ValueError(
            "specific_map must contain -1 or declared specific indices fitting int32"
        )
    ap = _real_array(a_primary, "a_primary", (items, p))
    asp = _real_array(a_specific, "a_specific", (items,))
    d = _real_array(threshold, "threshold", (items, n_cat - 1))
    mu = _real_array(latent_mean, "latent_mean", (p + s,))
    sd = _real_array(latent_sd, "latent_sd", (p + s,))
    if (sd <= 0).any():
        raise ValueError("latent_sd must be positive")
    if (np.diff(d, axis=1) >= 0).any():
        raise ValueError("threshold must be strictly decreasing per item")
    if (ap[~pm] != 0).any() or (asp[sm == -1] != 0).any():
        raise ValueError("fixed pattern positions must have zero slopes")
    args = (
        y.reshape(-1),
        np.ascontiguousarray(observed).reshape(-1),
        np.ascontiguousarray(pm).reshape(-1),
        np.ascontiguousarray(sm, dtype=np.int64),
        ap.reshape(-1),
        asp,
        d.reshape(-1),
        mu,
        sd,
        n,
        items,
        p,
        s,
        n_cat,
        qp,
        qs,
    )
    return args, (n, p + s)


def _person_scores(result, shape, gpu_memory_budget_bytes=None):
    """Reshape native Cai (2010), Appendix B moments without recomputation."""
    return TwoTierGrmPersonScores(
        *(
            np.asarray(result[key], dtype=np.float64).reshape(shape)
            for key in ("person_mean", "person_second", "person_sd")
        ),
        float(result["loglik"]),
        str(result.get("backend", "cpu")),
        gpu_memory_budget_bytes,
    )


def score_two_tier_grm_orthogonal(
    responses,
    primary_map,
    specific_map,
    a_primary,
    a_specific,
    threshold,
    latent_mean,
    latent_sd,
    *,
    n_cat,
    n_primary,
    n_specific,
    q_primary,
    q_specific,
    device="cpu",
    gpu_memory_budget_bytes=None,
    cache_item_tables=False,
):
    """Score fixed items under declared independent Gaussian latent factors.

    Cai (2010), p. 587 eqs. 4-6; p. 609 Appendix B. Caller supplies all means,
    SDs and node counts. Arrays order primaries then specifics. Missing persons
    retain the quadrature prior. Accuracy requires node sensitivity evidence.
    Explicit device="gpu" uses existing WGSL f32 posterior kernels and Rust
    f64 moment contraction and f64 host likelihood certification
    (Cai, 2010, pp.608-609 Appendices A/B;
    https://www.w3.org/TR/WGSL/#floating-point-types). GPU failure raises;
    CPU software adapters and automatic CPU fallback are rejected. Numerical
    parity, convergence and integration sensitivity remain required.
    GPU requires positive gpu_memory_budget_bytes: a caller-owned total budget
    for simultaneous input, posterior, intermediate and readback buffers.
    wgpu 30 Limits constrain individual buffers, not physical VRAM:
    https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html. Driver overhead and competing allocations can still fail.
    cache_item_tables=True opts CPU into caching identical fixed item/node
    probabilities (Cai, 2010, p.588 eq.9; Appendices A/B). It trades
    O(items * primary_grid * specific_nodes * categories) f64 storage for
    avoiding repeated probability evaluations. Default CPU scoring streams;
    GPU always requires its tables. Tables rebuild after each density update.
    Competing allocations can still fail through Device::push_error_scope:
    https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.push_error_scope.
    """
    if not isinstance(device, str) or device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    if not isinstance(cache_item_tables, bool):
        raise ValueError("cache_item_tables must be boolean")
    if device == "gpu":
        gpu_memory_budget_bytes = _bounded_integer(
            gpu_memory_budget_bytes, "gpu_memory_budget_bytes", 1, 2**64 - 1
        )
    args, shape = _native_inputs(
        responses,
        primary_map,
        specific_map,
        a_primary,
        a_specific,
        threshold,
        latent_mean,
        latent_sd,
        n_cat=n_cat,
        n_primary=n_primary,
        n_specific=n_specific,
        q_primary=q_primary,
        q_specific=q_specific,
    )
    core = _core_module()
    if core is None or not hasattr(core, "score_two_tier_grm_orthogonal"):
        raise RuntimeError(
            "score_two_tier_grm_orthogonal requires the updated compiled Rust core"
        )
    result = (
        core.score_two_tier_grm_orthogonal(*args)
        if device == "cpu" and not cache_item_tables
        else core.score_two_tier_grm_orthogonal(
            *args, device=device, gpu_memory_budget_bytes=gpu_memory_budget_bytes,
            cache_item_tables=cache_item_tables
        )
    )
    return _person_scores(result, shape, gpu_memory_budget_bytes if device == "gpu" else None)


def fit_two_tier_grm_focal_orthogonal(
    responses,
    primary_map,
    specific_map,
    a_primary,
    a_specific,
    threshold,
    latent_mean,
    latent_sd,
    *,
    n_cat,
    n_primary,
    n_specific,
    q_primary,
    q_specific,
    max_iter,
    tol,
    device="cpu",
    gpu_memory_budget_bytes=None,
    cache_item_tables=False,
):
    """Fit all latent means/SDs with every item fixed and factors independent.

    Cai (2010), pp. 608-609 Appendix A; Gaussian EM derivation in Rust docs.
    latent_mean/sd are explicit initialization; no defaults or hidden restart.
    Rust returns final scores even on nonconvergence for diagnostic records.
    A likelihood decrease or update cap remains nonconvergence. Preserve the
    fixed bank/maps with this fit. Native fitting rejects an observed loading
    matrix that fails its necessary free-mean rank guard (derived from Cai,
    2010, p. 589 eq. 11; LAPACK DGETF2 Purpose/INFO and pivot loop,
    https://www.netlib.org/lapack/double/dgetf2.f). The column-normalized
    max(rows, columns)*machine-epsilon threshold is an implementation choice,
    not a source-prescribed scientific cutoff or DGELSY effective rank.
    Passing does not prove variance/joint identification, population recovery
    or quadrature accuracy; these remain separate acceptance conditions.
    Explicit device="gpu" uses existing WGSL f32 posterior kernels and Rust
    f64 moment contraction and f64 host likelihood certification
    (Cai, 2010, pp.608-609 Appendices A/B;
    https://www.w3.org/TR/WGSL/#floating-point-types). GPU failure raises;
    CPU software adapters and automatic CPU fallback are rejected. Numerical
    parity, convergence and integration sensitivity remain required.
    GPU requires positive gpu_memory_budget_bytes: a caller-owned total budget
    for simultaneous input, posterior, intermediate and readback buffers.
    wgpu 30 Limits constrain individual buffers, not physical VRAM:
    https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html. Driver overhead and competing allocations can still fail.
    cache_item_tables=True opts CPU into caching identical fixed item/node
    probabilities (Cai, 2010, p.588 eq.9; Appendices A/B). It trades
    O(items * primary_grid * specific_nodes * categories) f64 storage for
    avoiding repeated probability evaluations. Default CPU scoring streams;
    GPU always requires its tables. Tables rebuild after each density update.
    Competing allocations can still fail through Device::push_error_scope:
    https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.push_error_scope.
    """
    cap = _bounded_integer(max_iter, "max_iter", 1, int(np.iinfo(np.uintp).max))
    tolerance = _positive_real(tol, "tol")
    if not isinstance(device, str) or device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    if not isinstance(cache_item_tables, bool):
        raise ValueError("cache_item_tables must be boolean")
    if device == "gpu":
        gpu_memory_budget_bytes = _bounded_integer(
            gpu_memory_budget_bytes, "gpu_memory_budget_bytes", 1, 2**64 - 1
        )
    args, shape = _native_inputs(
        responses,
        primary_map,
        specific_map,
        a_primary,
        a_specific,
        threshold,
        latent_mean,
        latent_sd,
        n_cat=n_cat,
        n_primary=n_primary,
        n_specific=n_specific,
        q_primary=q_primary,
        q_specific=q_specific,
    )
    core = _core_module()
    if core is None or not hasattr(core, "fit_two_tier_grm_focal_orthogonal"):
        raise RuntimeError(
            "fit_two_tier_grm_focal_orthogonal requires the updated compiled Rust core"
        )
    result = (
        core.fit_two_tier_grm_focal_orthogonal(*args, cap, tolerance)
        if device == "cpu" and not cache_item_tables
        else core.fit_two_tier_grm_focal_orthogonal(
            *args, cap, tolerance, device=device, gpu_memory_budget_bytes=gpu_memory_budget_bytes,
            cache_item_tables=cache_item_tables
        )
    )
    return TwoTierGrmFocalFit(
        np.asarray(result["latent_mean"], dtype=np.float64),
        np.asarray(result["latent_sd"], dtype=np.float64),
        _person_scores(result, shape, gpu_memory_budget_bytes if device == "gpu" else None),
        np.asarray(result["loglik_trace"], dtype=np.float64),
        int(result["n_iter"]),
        bool(result["converged"]),
        str(result["termination_reason"]),
        float(result["final_loglik_change"]),
        np.asarray(result["initial_mean"], dtype=np.float64),
        np.asarray(result["initial_sd"], dtype=np.float64),
        int(result["q_primary"]),
        int(result["q_specific"]),
        int(result["max_iter"]),
        float(result["tol"]),
    )
