"""Cluster-aware orchestration for nestedness metadata and Vuong comparison.

The numerical Vuong statistic is delegated to the Rust-backed
:func:`fast_mlsirm.fitstats.vuong_nonnested` kernel. This module adds safe
validation, stable cluster aggregation, AIC/BIC corrections, deterministic
cluster bootstrap intervals, and explicit indeterminate outcomes. It does not
claim to implement Vuong's separate weighted-chi-square distinguishability
hypothesis test; ``distinguishable`` is the documented numerical prerequisite
that the variance of casewise log-likelihood differences exceeds ``omega_tol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .fitstats import vuong_nonnested


class ModelRelation(str, Enum):
    """Declared mathematical relationship between two candidate models."""

    NESTED = "nested"
    STRICTLY_NON_NESTED = "strictly_non_nested"
    OVERLAPPING = "overlapping"
    BOUNDARY_NESTED = "boundary_nested"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelComparisonResult:
    """Result of a cluster-aware pairwise likelihood comparison.

    ``preferred_model`` is ``"a"``, ``"b"``, or ``None`` when the models are
    numerically indistinguishable or the two-sided Vuong test is not
    significant at ``alpha``. ``bootstrap_ci`` is a percentile interval for
    the corrected mean clusterwise log-likelihood difference.
    """

    relation: ModelRelation
    correction: str
    n_cases: int
    n_clusters: int
    mean_difference: float
    omega: float
    distinguishable: bool
    z: float
    p_two_sided: float
    preferred_model: str | None
    bootstrap_ci: tuple[float, float]
    warning: str | None


def _finite_vector(values, name: str) -> np.ndarray:
    """Return a finite non-empty one-dimensional float64 vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a 1-D vector with at least two entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_integer(value, name: str) -> int:
    """Validate and return a non-negative integer parameter count."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return converted


def _cluster_sums(a: np.ndarray, b: np.ndarray, cluster_id) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate paired casewise contributions into stable first-seen clusters."""
    if cluster_id is None:
        return a, b
    raw = np.asarray(cluster_id)
    if raw.ndim != 1 or raw.size != a.size:
        raise ValueError("cluster_id must be a 1-D vector matching the likelihood vectors")
    labels: dict[object, int] = {}
    compact = np.empty(raw.size, dtype=np.int64)
    for index, label in enumerate(raw.tolist()):
        try:
            compact[index] = labels.setdefault(label, len(labels))
        except TypeError as exc:
            raise ValueError("cluster_id entries must be hashable") from exc
    if len(labels) < 2:
        raise ValueError("cluster_id must identify at least two clusters")
    out_a = np.bincount(compact, weights=a, minlength=len(labels)).astype(np.float64)
    out_b = np.bincount(compact, weights=b, minlength=len(labels)).astype(np.float64)
    return out_a, out_b


def _corrected_inputs(
    a: np.ndarray,
    b: np.ndarray,
    k_a: int,
    k_b: int,
    correction: str,
) -> tuple[np.ndarray, bool]:
    """Return inputs and the native BIC switch for the requested correction."""
    normalized = str(correction).lower()
    if normalized == "none":
        return a, False
    if normalized == "bic":
        return a, True
    if normalized == "aic":
        shifted = a - (k_a - k_b) / a.size
        return shifted, False
    raise ValueError("correction must be one of ['none', 'aic', 'bic']")


def _bootstrap_interval(
    differences: np.ndarray,
    k_delta: int,
    correction: str,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """Percentile cluster-bootstrap interval for corrected mean difference."""
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)):
        raise ValueError("bootstrap must be an integer in 0..100000")
    replicates = int(replicates)
    if not 0 <= replicates <= 100_000:
        raise ValueError("bootstrap must be an integer in 0..100000")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    if replicates == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = differences.size
    penalty = 0.0
    if correction == "aic":
        penalty = k_delta / n
    elif correction == "bic":
        penalty = 0.5 * k_delta * np.log(n) / n
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, n, size=n)
        estimates[replicate] = float(np.mean(differences[sampled]) - penalty)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def compare_nonnested_models(
    loglik_a,
    loglik_b,
    k_a: int,
    k_b: int,
    *,
    relation: ModelRelation | str = ModelRelation.STRICTLY_NON_NESTED,
    cluster_id=None,
    correction: str = "bic",
    alpha: float = 0.05,
    omega_tol: float = 1e-12,
    bootstrap: int = 2000,
    seed: int = 1,
    confidence: float = 0.95,
) -> ModelComparisonResult:
    """Compare two fitted models from paired casewise marginal log-likelihoods.

    Cases can be aggregated by query, system, judge family, or another
    independent sampling unit through ``cluster_id``. Positive differences and
    positive z values favor model A. For ``boundary_nested`` or ``nested``
    declarations, the result is descriptive and emits a warning because a
    boundary-aware or ordinary likelihood-ratio procedure may be preferable.

    References
    ----------
    Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020).
    Model selection of nested and non-nested item response models using Vuong
    tests. *Multivariate Behavioral Research, 55*(5), 664-684.
    https://doi.org/10.1080/00273171.2019.1664280
    """
    a = _finite_vector(loglik_a, "loglik_a")
    b = _finite_vector(loglik_b, "loglik_b")
    if a.shape != b.shape:
        raise ValueError("loglik_a and loglik_b must have the same length")
    k_a = _nonnegative_integer(k_a, "k_a")
    k_b = _nonnegative_integer(k_b, "k_b")
    try:
        relation_value = relation if isinstance(relation, ModelRelation) else ModelRelation(str(relation))
    except ValueError as exc:
        raise ValueError(f"relation must be one of {[item.value for item in ModelRelation]}") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not np.isfinite(omega_tol) or omega_tol < 0.0:
        raise ValueError("omega_tol must be finite and non-negative")
    clustered_a, clustered_b = _cluster_sums(a, b, cluster_id)
    normalized_correction = str(correction).lower()
    corrected_a, native_bic = _corrected_inputs(
        clustered_a, clustered_b, k_a, k_b, normalized_correction
    )
    raw_difference = clustered_a - clustered_b
    raw_omega = float(np.std(raw_difference, ddof=0))
    distinguishable = bool(raw_omega > omega_tol)
    warning = None
    if relation_value in {ModelRelation.NESTED, ModelRelation.BOUNDARY_NESTED}:
        warning = (
            "Declared relation is nested or boundary-nested; use an ordinary or "
            "parametric-bootstrap likelihood-ratio test as the primary procedure."
        )
    if not distinguishable:
        return ModelComparisonResult(
            relation=relation_value,
            correction=normalized_correction,
            n_cases=int(a.size),
            n_clusters=int(clustered_a.size),
            mean_difference=float(np.mean(raw_difference)),
            omega=raw_omega,
            distinguishable=False,
            z=float("nan"),
            p_two_sided=float("nan"),
            preferred_model=None,
            bootstrap_ci=(float("nan"), float("nan")),
            warning="Models are observationally indistinguishable at omega_tol.",
        )
    statistic = vuong_nonnested(
        corrected_a,
        clustered_b,
        k_a,
        k_b,
        bic_correction=native_bic,
    )
    z = float(statistic["z"])
    p_value = float(statistic["p_two_sided"])
    preferred = None if p_value >= alpha else ("a" if z > 0.0 else "b")
    interval = _bootstrap_interval(
        raw_difference,
        k_a - k_b,
        normalized_correction,
        bootstrap,
        seed,
        confidence,
    )
    corrected_mean = float(statistic["mean_diff"])
    if normalized_correction == "bic":
        corrected_mean -= 0.5 * (k_a - k_b) * np.log(clustered_a.size) / clustered_a.size
    return ModelComparisonResult(
        relation=relation_value,
        correction=normalized_correction,
        n_cases=int(a.size),
        n_clusters=int(clustered_a.size),
        mean_difference=corrected_mean,
        omega=float(statistic["omega"]),
        distinguishable=True,
        z=z,
        p_two_sided=p_value,
        preferred_model=preferred,
        bootstrap_ci=interval,
        warning=warning,
    )
