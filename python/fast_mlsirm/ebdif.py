"""Empirical Bayes Mantel-Haenszel DIF. All numeric work happens in the
Rust core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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
_EBDIF_RESULT_KEYS = frozenset(
    {"mu", "tau2", "tau2_raw", "weight", "post_mean", "post_var", "cat_probs"}
)
_MAX_EBDIF_ITEMS = 20_000_000
_EBDIF_RESULT_VALUES_PER_ITEM = 8
_MAX_EBDIF_RESULT_VALUES = 20_000_000
_RESULT_FINITE_CHUNK = 65_536


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


def _enforce_result_budget(n_items: int) -> None:
    """Bound deterministic native-plus-snapshot output from inert cardinality."""

    if n_items > _MAX_EBDIF_RESULT_VALUES // _EBDIF_RESULT_VALUES_PER_ITEM:
        raise ValueError(
            f"EBDIF result exceeds the {_MAX_EBDIF_RESULT_VALUES}-value resource limit"
        )


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


def _validated_1d(x, name: str, *, expected_length: int) -> np.ndarray:
    """Seal and validate bounded real 1-D evidence without caller-owned aliases."""

    if type(x) is np.ndarray:
        if x.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        current_length = int(x.shape[0])
        _enforce_item_budget(name, current_length)
        if current_length != expected_length:
            raise ValueError("mh and se must have the same length")
        if x.dtype.kind == "c":
            raise ValueError(f"{name} must be real-valued")
        if x.dtype.kind not in ("b", "i", "u", "f"):
            raise ValueError(f"{name} must be a numeric array")
        try:
            xa = np.array(x, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
    elif type(x) in (list, tuple):
        current_length = len(x)
        _enforce_item_budget(name, current_length)
        if current_length != expected_length:
            raise ValueError("mh and se must have the same length")
        snapshot = list.copy(x) if type(x) is list else x
        if len(snapshot) != expected_length:
            raise ValueError("mh and se must have the same length")
        if any(type(value) not in _TRUSTED_EBDIF_SCALAR_TYPES for value in snapshot):
            raise ValueError(f"{name} must be a numeric array")
        if any(not _scalar_preserves_float64_identity(value) for value in snapshot):
            raise ValueError(f"{name} entries must be exactly representable as float64")
        try:
            xa = np.asarray(snapshot)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
    else:
        raise ValueError(f"{name} must be a numeric 1-D array")

    if xa.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if int(xa.shape[0]) != expected_length:
        raise ValueError("mh and se must have the same length")
    if np.iscomplexobj(xa):
        raise ValueError(f"{name} must be real-valued")
    if xa.dtype.kind == "b":
        xa = xa.astype(np.float64)
    if xa.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    return _lossless_float64_array(xa, name)


def _exact_finite_result_vector(
    value: object,
    *,
    expected_length: int,
    invalid: RuntimeError,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    """Seal one concrete PyO3 float64 vector without conversion protocols."""

    if type(value) is not np.ndarray:
        raise invalid
    if value.dtype != np.dtype(np.float64) or value.ndim != 1:
        raise invalid
    if int(value.shape[0]) != expected_length:
        raise invalid
    if not value.flags.c_contiguous or not value.flags.owndata:
        raise invalid

    snapshot = value.copy(order="C")
    if type(snapshot) is not np.ndarray:
        raise invalid
    if snapshot.dtype != np.dtype(np.float64) or snapshot.ndim != 1:
        raise invalid
    if int(snapshot.shape[0]) != expected_length:
        raise invalid
    if not snapshot.flags.c_contiguous or not snapshot.flags.owndata:
        raise invalid

    for start in range(0, expected_length, _RESULT_FINITE_CHUNK):
        stop = min(start + _RESULT_FINITE_CHUNK, expected_length)
        chunk = snapshot[start:stop]
        if not bool(np.isfinite(chunk).all()):
            raise invalid
        if minimum is not None and not bool((chunk >= minimum).all()):
            raise invalid
        if maximum is not None and not bool((chunk <= maximum).all()):
            raise invalid
    return snapshot


def _validated_rust_result(
    result: object,
    *,
    n_items: int,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the concrete PyO3 EBDIF result envelope before public marshalling."""

    invalid = RuntimeError("invalid EBDIF Rust result payload")
    if type(result) is not dict or len(result) != len(_EBDIF_RESULT_KEYS):
        raise invalid

    root = dict.copy(result)
    if type(root) is not dict or len(root) != len(_EBDIF_RESULT_KEYS):
        raise invalid
    keys = list(dict.keys(root))
    if any(type(key) is not str for key in keys) or set(keys) != _EBDIF_RESULT_KEYS:
        raise invalid

    mu = dict.__getitem__(root, "mu")
    tau2 = dict.__getitem__(root, "tau2")
    tau2_raw = dict.__getitem__(root, "tau2_raw")
    if any(type(value) is not float or not isfinite(value) for value in (mu, tau2, tau2_raw)):
        raise invalid
    expected_tau2 = tau2_raw if tau2_raw > 0.0 else 0.0
    if tau2 != expected_tau2:
        raise invalid

    weight = _exact_finite_result_vector(
        dict.__getitem__(root, "weight"),
        expected_length=n_items,
        invalid=invalid,
        minimum=0.0,
        maximum=1.0,
    )
    post_mean = _exact_finite_result_vector(
        dict.__getitem__(root, "post_mean"),
        expected_length=n_items,
        invalid=invalid,
    )
    post_var = _exact_finite_result_vector(
        dict.__getitem__(root, "post_var"),
        expected_length=n_items,
        invalid=invalid,
        minimum=0.0,
    )
    cat_probs = _exact_finite_result_vector(
        dict.__getitem__(root, "cat_probs"),
        expected_length=n_items * 5,
        invalid=invalid,
        minimum=0.0,
        maximum=1.0,
    )
    return mu, tau2, tau2_raw, weight, post_mean, post_var, cat_probs


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
    _enforce_result_budget(mh_length)

    mhf = _validated_1d(mh, "mh", expected_length=mh_length)
    sef = _validated_1d(se, "se", expected_length=se_length)
    if not np.all(np.isfinite(mhf)):
        raise ValueError("mh entries must be finite (filter NaN MH rows first)")
    if not np.all(np.isfinite(sef) & (sef > 0.0)):
        raise ValueError("se entries must be finite and positive")

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_eb_mh_dif"):
        raise RuntimeError("Rust core with py_eb_mh_dif is required")
    res = core.py_eb_mh_dif(mhf, sef)
    ni = int(mhf.shape[0])
    mu, tau2, tau2_raw, weight, post_mean, post_var, cat_probs = _validated_rust_result(
        res,
        n_items=ni,
    )
    return EbDifResult(
        mu=mu,
        tau2=tau2,
        tau2_raw=tau2_raw,
        weight=weight,
        post_mean=post_mean,
        post_var=post_var,
        cat_probs=cat_probs.reshape(ni, 5),
    )
