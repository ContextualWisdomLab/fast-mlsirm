"""Empirical Bayes Mantel-Haenszel DIF. All numeric work happens in the
Rust core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TRUSTED_EBDIF_SCALAR_TYPES = frozenset(
    {
        bool,
        int,
        float,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    }
)
_TRUSTED_EBDIF_INTEGER_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    }
)
_MAX_EBDIF_ITEMS = 20_000_000


@dataclass
class EbDifResult:
    """Empirical Bayes MH DIF result.

    ``mu`` and ``tau2`` are the estimated prior mean/variance (``tau2``
    floored at 0; ``tau2_raw`` is the pre-floor diagnostic and the only
    place a negative estimate is visible). ``weight`` holds the shrinkage
    weights ``W_i = tau2 / (tau2 + se_i**2)``; ``post_mean`` the EB point
    estimates ``W_i * mh_i + (1 - W_i) * mu``; ``post_var`` the posterior
    variances ``W_i * se_i**2``; ``cat_probs`` the ``n_items x 5``
    posterior probabilities of the ETS DIF categories, columns ordered
    ``[C-, B-, A, B+, C+]``."""

    mu: float
    tau2: float
    tau2_raw: float
    weight: np.ndarray
    post_mean: np.ndarray
    post_var: np.ndarray
    cat_probs: np.ndarray


def _enforce_item_budget(name: str, length: int) -> None:
    """Bound item evidence before package-owned dense float64 allocation."""

    if length > _MAX_EBDIF_ITEMS:
        raise ValueError(f"{name} exceeds the {_MAX_EBDIF_ITEMS}-item resource limit")


def _trusted_1d_length(x, name: str) -> int:
    """Return bounded 1-D carrier length after callback-free storage checks."""

    if type(x) is np.ndarray:
        if x.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        length = int(x.shape[0])
        _enforce_item_budget(name, length)
        if x.dtype.kind == "c":
            raise ValueError(f"{name} must be real-valued")
        if x.dtype.kind not in ("b", "i", "u", "f"):
            raise ValueError(f"{name} must be a numeric array")
        return length
    if type(x) in (list, tuple):
        length = len(x)
        _enforce_item_budget(name, length)
        if any(type(value) not in _TRUSTED_EBDIF_SCALAR_TYPES for value in x):
            raise ValueError(f"{name} must be a numeric array")
        return length
    raise ValueError(f"{name} must be a numeric 1-D array")


def _scalar_preserves_float64_identity(value: object) -> bool:
    """Return whether one trusted scalar survives exact Rust-f64 normalization."""

    value_type = type(value)
    if value_type in (bool, np.bool_):
        return True
    if value_type in _TRUSTED_EBDIF_INTEGER_TYPES:
        exact = int(value)
        try:
            narrowed = float(exact)
        except OverflowError:
            return False
        return np.isfinite(narrowed) and int(narrowed) == exact
    if value_type in (float, np.float16, np.float32, np.float64):
        return True
    if value_type is np.longdouble:
        if not np.isfinite(value):
            return True
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.float64(value)
        return bool(np.isfinite(narrowed) and np.longdouble(narrowed) == value)
    return False


def _lossless_float64_array(xa: np.ndarray, name: str) -> np.ndarray:
    """Normalize trusted numeric evidence without changing any finite value."""

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.ascontiguousarray(xa, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc

    if xa.dtype.kind in ("i", "u") and xa.dtype.itemsize > 4:
        with np.errstate(over="ignore", invalid="ignore"):
            roundtrip = narrowed.astype(xa.dtype)
        if not np.array_equal(roundtrip, xa):
            raise ValueError(f"{name} entries must be exactly representable as float64")
    elif xa.dtype.kind == "f" and xa.dtype.itemsize > np.dtype(np.float64).itemsize:
        finite = np.isfinite(xa)
        with np.errstate(over="ignore", invalid="ignore"):
            roundtrip = narrowed.astype(xa.dtype)
        if not np.array_equal(roundtrip[finite], xa[finite]):
            raise ValueError(f"{name} entries must be exactly representable as float64")

    return narrowed


def _validated_1d(x, name: str) -> np.ndarray:
    """Validate trusted, resource-bounded real 1-D evidence without caller hooks."""
    if type(x) is np.ndarray:
        if x.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        _enforce_item_budget(name, int(x.shape[0]))
        xa = x
    elif type(x) in (list, tuple):
        _enforce_item_budget(name, len(x))
        if any(type(value) not in _TRUSTED_EBDIF_SCALAR_TYPES for value in x):
            raise ValueError(f"{name} must be a numeric array")
        if any(not _scalar_preserves_float64_identity(value) for value in x):
            raise ValueError(f"{name} entries must be exactly representable as float64")
        try:
            xa = np.asarray(x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
    else:
        raise ValueError(f"{name} must be a numeric 1-D array")

    if xa.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if np.iscomplexobj(xa):
        raise ValueError(f"{name} must be real-valued")
    if xa.dtype.kind == "b":
        xa = xa.astype(np.float64)
    if xa.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    return _lossless_float64_array(xa, name)


def eb_mh_dif(mh, se) -> EbDifResult:
    """Empirical Bayes Mantel-Haenszel DIF
    (``mlsirm_core::dif::eb_mh_dif``; Zwick & Thayer, 2003, ERIC
    ED481063 — READ; the statistical-model section, report pp. 3-5,
    supplies every formula). NOT READ (history only, no formulas taken):
    Zwick, Thayer & Lewis (1999, 1997); Longford, Holland & Thayer
    (1993).

    ``mh`` holds per-item MH D-DIF statistics on the ETS delta scale and
    ``se`` their standard errors — e.g. ``MhDifRow.mh_d_dif`` /
    ``se_d_dif`` from :func:`fast_mlsirm.mantel_haenszel_dif`; filter out
    items whose MH statistics are NaN before calling. The prior
    ``N(mu, tau2)`` is estimated from exactly the supplied item set
    (``mu`` = mean, ``tau2`` = sample variance (divisor ``n-1``, an
    implementation choice — the paper does not print the divisor) minus
    the mean squared SE, floored at 0 per the report's footnote 5). Each
    item's posterior is normal with mean ``W*mh + (1-W)*mu`` and variance
    ``W*se**2`` where ``W = tau2/(tau2 + se**2)``; the five-category
    probabilities are its areas over the ETS bands delimited at -1.5,
    -1, 1, 1.5. When ``tau2 == 0`` the posterior is a point mass at
    ``mu`` and the category row is an indicator (boundary conventions
    implementation-defined).

    In LLM-as-a-Judge quality management this stabilizes small-sample
    MH DIF screening of evaluation items by shrinking noisy per-item
    DIF estimates toward the pool mean.

    References
    ----------
    Zwick, R., & Thayer, D. T. (2003). *An empirical Bayes enhancement
    of Mantel-Haenszel DIF analysis for computer-adaptive tests* (LSAC
    Research Report Series; ERIC ED481063).
    """
    mh_length = _trusted_1d_length(mh, "mh")
    se_length = _trusted_1d_length(se, "se")
    if mh_length != se_length:
        raise ValueError("mh and se must have the same length")
    if mh_length < 2:
        raise ValueError("need at least 2 items")

    mhf = _validated_1d(mh, "mh")
    sef = _validated_1d(se, "se")
    if not np.all(np.isfinite(mhf)):
        raise ValueError("mh entries must be finite (filter NaN MH rows first)")
    if not np.all(np.isfinite(sef) & (sef > 0.0)):
        raise ValueError("se entries must be finite and positive")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_eb_mh_dif"):
        raise RuntimeError("Rust core with py_eb_mh_dif is required")
    res = core.py_eb_mh_dif(mhf, sef)
    ni = mhf.shape[0]
    return EbDifResult(
        mu=float(res["mu"]),
        tau2=float(res["tau2"]),
        tau2_raw=float(res["tau2_raw"]),
        weight=np.asarray(res["weight"], dtype=np.float64),
        post_mean=np.asarray(res["post_mean"], dtype=np.float64),
        post_var=np.asarray(res["post_var"], dtype=np.float64),
        cat_probs=np.asarray(res["cat_probs"], dtype=np.float64).reshape(ni, 5),
    )
