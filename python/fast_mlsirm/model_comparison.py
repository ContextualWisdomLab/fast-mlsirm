"""Fail-closed orchestration for Rust-backed Vuong selection statistics.

All statistical quantities are computed by
:func:`fast_mlsirm.fitstats.vuong_nonnested`, whose implementation resides in
the compiled Rust core. This module validates bounded audit metadata and
prevents a non-nested normal selection statistic from being interpreted as a
model preference before Vuong's formal distinguishability test is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import islice
import math
import operator
from typing import Any

from .fitstats import vuong_nonnested

MAX_CASEWISE_VALUES = 1_000_000
MAX_MODEL_LABEL_CHARS = 128


class ModelRelation(str, Enum):
    """Declared mathematical relationship between two candidate models."""

    NESTED = "nested"
    STRICTLY_NON_NESTED = "strictly_non_nested"
    OVERLAPPING = "overlapping"
    BOUNDARY_NESTED = "boundary_nested"
    UNKNOWN = "unknown"


class ComparisonStatus(str, Enum):
    """Fail-closed interpretation status for a model-selection summary."""

    VARIANCE_DEGENERATE = "variance_degenerate"
    REQUIRES_LIKELIHOOD_RATIO = "requires_likelihood_ratio"
    REQUIRES_DISTINGUISHABILITY_TEST = "requires_distinguishability_test"
    UNKNOWN_RELATION = "unknown_relation"


class VuongVarianceDegenerateError(ValueError):
    """Typed boundary signal for exact zero variance in casewise differences."""


@dataclass(frozen=True)
class ModelComparisonResult:
    """Auditable result of one pairwise likelihood-selection calculation.

    ``raw_mean_loglik_difference``, ``omega``, ``raw_z``, and
    ``raw_p_two_sided`` preserve values returned by the Rust selection kernel.
    ``z`` and ``p_two_sided`` remain unavailable until the mathematically
    required relation-specific precondition has been satisfied. In particular,
    positive sample variance is a numerical stability condition and is not
    Vuong's formal weighted-chi-square distinguishability test.
    """

    model_a: str
    model_b: str
    relation: ModelRelation
    status: ComparisonStatus
    bic_correction: bool
    n_cases: int
    k_a: int
    k_b: int
    raw_mean_loglik_difference: float
    omega: float
    variance_positive: bool
    raw_z: float
    raw_p_two_sided: float
    z: float
    p_two_sided: float
    preferred_model: str | None
    warning: str


def _model_label(value: str, name: str) -> str:
    """Return a bounded printable model label suitable for audit output."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    if len(normalized) > MAX_MODEL_LABEL_CHARS:
        raise ValueError(
            f"{name} must contain at most {MAX_MODEL_LABEL_CHARS} characters"
        )
    if not normalized.isprintable():
        raise ValueError(f"{name} must not contain control characters")
    return normalized


def _relation(value: ModelRelation | str) -> ModelRelation:
    """Normalize a relation value or raise an explicit validation error."""
    if isinstance(value, ModelRelation):
        return value
    try:
        return ModelRelation(str(value))
    except ValueError as exc:
        choices = [item.value for item in ModelRelation]
        raise ValueError(f"relation must be one of {choices}") from exc


def _parameter_count(value: Any, name: str) -> int:
    """Return a non-negative integer parameter count while rejecting booleans."""
    is_numpy_boolean = (
        value.__class__.__module__.startswith("numpy")
        and value.__class__.__name__ == "bool_"
    )
    if isinstance(value, bool) or is_numpy_boolean:
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(normalized)


def _casewise_values(value: Any, name: str) -> tuple[Any, ...]:
    """Materialize a bounded casewise iterable with stable public errors."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of numeric casewise values")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an iterable of numeric casewise values"
        ) from exc
    materialized = tuple(islice(iterator, MAX_CASEWISE_VALUES + 1))
    if len(materialized) > MAX_CASEWISE_VALUES:
        raise ValueError(
            f"{name} must contain at most {MAX_CASEWISE_VALUES} casewise values"
        )
    return materialized


def _run_vuong(
    values_a: tuple[Any, ...],
    values_b: tuple[Any, ...],
    k_a: int,
    k_b: int,
    *,
    bic_correction: bool,
) -> dict[str, Any]:
    """Call the trusted kernel and translate its legacy zero-variance signal."""
    try:
        return vuong_nonnested(
            values_a,
            values_b,
            k_a,
            k_b,
            bic_correction=bic_correction,
        )
    except ValueError as exc:
        if "omega^2 = 0" not in str(exc):
            raise
        raise VuongVarianceDegenerateError(
            "casewise log-likelihood differences have exact zero variance"
        ) from exc


def _relation_requirement(
    relation: ModelRelation,
) -> tuple[ComparisonStatus, str]:
    """Return the procedure required before any model preference is reported."""
    if relation in {ModelRelation.NESTED, ModelRelation.BOUNDARY_NESTED}:
        return (
            ComparisonStatus.REQUIRES_LIKELIHOOD_RATIO,
            "Nested or boundary-nested models require an ordinary, mixture, or "
            "parametric-bootstrap likelihood-ratio procedure; the Vuong normal "
            "selection statistic is not interpreted as a preference.",
        )
    if relation in {
        ModelRelation.STRICTLY_NON_NESTED,
        ModelRelation.OVERLAPPING,
    }:
        return (
            ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST,
            "Non-nested selection requires Vuong's formal weighted-chi-square "
            "distinguishability test before the normal selection statistic can "
            "support a preference. Positive sample variance is not that test.",
        )
    return (
        ComparisonStatus.UNKNOWN_RELATION,
        "Model relation is unknown; establish nestedness or overlap before "
        "interpreting any likelihood-selection statistic.",
    )


def compare_nonnested_models(
    loglik_a: Any,
    loglik_b: Any,
    k_a: int,
    k_b: int,
    *,
    model_a: str = "A",
    model_b: str = "B",
    relation: ModelRelation | str = ModelRelation.UNKNOWN,
    bic_correction: bool = True,
    alpha: float = 0.05,
    omega_tol: float = 1e-12,
) -> ModelComparisonResult:
    """Summarize paired casewise marginal log-likelihood contributions.

    The Rust core computes the casewise log-likelihood-ratio mean, population
    standard deviation ``omega``, BIC-corrected or uncorrected Vuong z
    statistic, and two-sided normal p value. This wrapper deliberately returns
    no preferred model until the relation-appropriate inferential prerequisite
    has been established.

    Parameters
    ----------
    loglik_a, loglik_b:
        Paired casewise marginal log-likelihood contributions for models A and
        B. At most :data:`MAX_CASEWISE_VALUES` values are accepted per model.
    k_a, k_b:
        Non-negative free-parameter counts.
    model_a, model_b:
        Distinct bounded printable labels copied into the audit result.
    relation:
        Mathematical relationship between the fitted model families. The
        fail-closed default is ``unknown``.
    bic_correction:
        Apply Vuong's BIC penalty ``0.5 * (k_a-k_b) * log(n)`` to the summed
        log-likelihood ratio before standardization.
    alpha:
        Reserved two-sided selection threshold, validated for forward
        compatibility. It does not authorize preference without the formal
        distinguishability stage.
    omega_tol:
        Numerical variance floor. This is a stability guard, not the formal
        distinguishability hypothesis test.

    Returns
    -------
    ModelComparisonResult
        Rust-computed audit statistics and the required next statistical
        procedure. ``preferred_model`` is always ``None`` in this release.

    References
    ----------
    Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
    non-nested hypotheses. *Econometrica, 57*(2), 307-333.
    https://doi.org/10.2307/1912557

    Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020).
    Model selection of nested and non-nested item response models using Vuong
    tests. *Multivariate Behavioral Research, 55*(5), 664-684.
    https://doi.org/10.1080/00273171.2019.1664280
    """
    label_a = _model_label(model_a, "model_a")
    label_b = _model_label(model_b, "model_b")
    if label_a == label_b:
        raise ValueError("model_a and model_b must be distinct")
    relation_value = _relation(relation)
    count_a = _parameter_count(k_a, "k_a")
    count_b = _parameter_count(k_b, "k_b")
    if not isinstance(bic_correction, bool):
        raise ValueError("bic_correction must be boolean")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not math.isfinite(omega_tol) or omega_tol < 0.0:
        raise ValueError("omega_tol must be finite and non-negative")

    values_a = _casewise_values(loglik_a, "loglik_a")
    values_b = _casewise_values(loglik_b, "loglik_b")
    try:
        statistic = _run_vuong(
            values_a,
            values_b,
            count_a,
            count_b,
            bic_correction=bic_correction,
        )
    except VuongVarianceDegenerateError:
        raw_mean = raw_z = raw_p = float("nan")
        omega = 0.0
        variance_positive = False
        status = ComparisonStatus.VARIANCE_DEGENERATE
        warning = (
            "Casewise log-likelihood differences have exact zero variance; "
            "selection inference is undefined."
        )
    else:
        raw_mean = float(statistic["mean_diff"])
        omega = float(statistic["omega"])
        raw_z = float(statistic["z"])
        raw_p = float(statistic["p_two_sided"])
        variance_positive = math.isfinite(omega) and omega > omega_tol
        if (
            not variance_positive
            or not math.isfinite(raw_z)
            or not math.isfinite(raw_p)
        ):
            status = ComparisonStatus.VARIANCE_DEGENERATE
            warning = (
                "Casewise log-likelihood differences have zero, non-finite, or "
                "numerically degenerate variance; selection inference is undefined."
            )
        else:
            status, warning = _relation_requirement(relation_value)

    return ModelComparisonResult(
        model_a=label_a,
        model_b=label_b,
        relation=relation_value,
        status=status,
        bic_correction=bic_correction,
        n_cases=len(values_a),
        k_a=count_a,
        k_b=count_b,
        raw_mean_loglik_difference=raw_mean,
        omega=omega,
        variance_positive=variance_positive,
        raw_z=raw_z,
        raw_p_two_sided=raw_p,
        z=float("nan"),
        p_two_sided=float("nan"),
        preferred_model=None,
        warning=warning,
    )
