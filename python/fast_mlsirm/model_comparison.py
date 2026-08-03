"""Decision-safe orchestration for Rust-backed Vuong model comparison.

All statistical quantities are computed by
:func:`fast_mlsirm.fitstats.vuong_nonnested`, whose implementation resides in
the compiled Rust core. This module adds model-relation metadata and a
fail-closed interpretation layer so nested, boundary-nested, overlapping, and
unknown relationships are not silently reported with an invalid normal-theory
preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .fitstats import vuong_nonnested


class ModelRelation(str, Enum):
    """Declared mathematical relationship between two candidate models."""

    NESTED = "nested"
    STRICTLY_NON_NESTED = "strictly_non_nested"
    OVERLAPPING = "overlapping"
    BOUNDARY_NESTED = "boundary_nested"
    UNKNOWN = "unknown"


class ComparisonStatus(str, Enum):
    """Interpretation status returned by :func:`compare_nonnested_models`."""

    MODEL_A_PREFERRED = "model_a_preferred"
    MODEL_B_PREFERRED = "model_b_preferred"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    VARIANCE_DEGENERATE = "variance_degenerate"
    REQUIRES_LIKELIHOOD_RATIO = "requires_likelihood_ratio"
    REQUIRES_DISTINGUISHABILITY_TEST = "requires_distinguishability_test"
    UNKNOWN_RELATION = "unknown_relation"


@dataclass(frozen=True)
class ModelComparisonResult:
    """Auditable result of a pairwise likelihood comparison.

    ``raw_mean_loglik_difference`` and ``omega`` preserve Rust-kernel values
    whenever the kernel can form them. For an exact zero-variance comparison,
    the strict low-level Rust API reports its dedicated indistinguishability
    error; the decision wrapper represents that state with ``omega = 0`` and
    ``raw_mean_loglik_difference = NaN`` rather than reproducing the statistic
    in Python. ``z`` and ``p_two_sided`` are exposed only when the declared
    relationship is strictly non-nested and the casewise difference variance
    exceeds ``omega_tol``. Positive differences favor ``model_a``.
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
    z: float
    p_two_sided: float
    preferred_model: str | None
    warning: str | None


def _model_label(value: str, name: str) -> str:
    """Return a non-empty model label suitable for audit output."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relation(value: ModelRelation | str) -> ModelRelation:
    """Normalize a relation value or raise an explicit validation error."""
    if isinstance(value, ModelRelation):
        return value
    try:
        return ModelRelation(str(value))
    except ValueError as exc:
        choices = [item.value for item in ModelRelation]
        raise ValueError(f"relation must be one of {choices}") from exc


def _casewise_values(value, name: str) -> tuple:
    """Materialize one casewise iterable with a stable public validation error."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of numeric casewise values")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an iterable of numeric casewise values"
        ) from exc


def _unsupported_relation(
    relation: ModelRelation,
) -> tuple[ComparisonStatus, str] | None:
    """Return the fail-closed status and warning for unsupported relations."""
    if relation in {ModelRelation.NESTED, ModelRelation.BOUNDARY_NESTED}:
        return (
            ComparisonStatus.REQUIRES_LIKELIHOOD_RATIO,
            "Nested or boundary-nested models require an ordinary, mixture, or "
            "parametric-bootstrap likelihood-ratio procedure; the Vuong normal "
            "preference is intentionally suppressed.",
        )
    if relation is ModelRelation.OVERLAPPING:
        return (
            ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST,
            "Overlapping models require Vuong's formal weighted-chi-square "
            "distinguishability test before normal-theory preference inference; "
            "that test requires casewise scores and information matrices.",
        )
    if relation is ModelRelation.UNKNOWN:
        return (
            ComparisonStatus.UNKNOWN_RELATION,
            "Model relation is unknown; establish nestedness or overlap before "
            "interpreting a Vuong preference.",
        )
    return None


def compare_nonnested_models(
    loglik_a,
    loglik_b,
    k_a: int,
    k_b: int,
    *,
    model_a: str = "A",
    model_b: str = "B",
    relation: ModelRelation | str = ModelRelation.STRICTLY_NON_NESTED,
    bic_correction: bool = True,
    alpha: float = 0.05,
    omega_tol: float = 1e-12,
) -> ModelComparisonResult:
    """Compare paired casewise marginal log-likelihood contributions.

    The Rust core computes the casewise log-likelihood-ratio mean, population
    standard deviation ``omega``, BIC-corrected or uncorrected Vuong z statistic,
    and two-sided normal p value. This wrapper interprets those values only for
    a declared strictly non-nested comparison. It deliberately does not label
    ``omega > omega_tol`` as Vuong's formal distinguishability test. Exact
    zero-variance comparisons are converted from the low-level Rust
    indistinguishability error into a non-preference result rather than leaking
    an exception through the decision API.

    Parameters
    ----------
    loglik_a, loglik_b:
        Paired casewise marginal log-likelihood contributions for models A and
        B. The low-level Rust-backed wrapper validates shape and finiteness.
    k_a, k_b:
        Non-negative free-parameter counts.
    model_a, model_b:
        Distinct labels copied into the auditable result.
    relation:
        Mathematical relationship between the fitted model families.
    bic_correction:
        Apply Vuong's BIC penalty ``0.5 * (k_a-k_b) * log(n)`` to the summed
        log-likelihood ratio before standardization.
    alpha:
        Two-sided preference threshold, strictly between zero and one.
    omega_tol:
        Numerical variance floor. This is a stability guard, not the formal
        distinguishability hypothesis test.

    Returns
    -------
    ModelComparisonResult
        A fail-closed interpretation plus the Rust-computed audit statistics.

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
    if not isinstance(bic_correction, bool):
        raise ValueError("bic_correction must be boolean")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not math.isfinite(omega_tol) or omega_tol < 0.0:
        raise ValueError("omega_tol must be finite and non-negative")

    values_a = _casewise_values(loglik_a, "loglik_a")
    values_b = _casewise_values(loglik_b, "loglik_b")
    try:
        statistic = vuong_nonnested(
            values_a,
            values_b,
            k_a,
            k_b,
            bic_correction=bic_correction,
        )
    except ValueError as exc:
        if "omega^2 = 0" not in str(exc):
            raise
        raw_mean = raw_z = raw_p = float("nan")
        omega = 0.0
    else:
        raw_mean = float(statistic["mean_diff"])
        omega = float(statistic["omega"])
        raw_z = float(statistic["z"])
        raw_p = float(statistic["p_two_sided"])
    variance_positive = math.isfinite(omega) and omega > omega_tol

    unsupported = _unsupported_relation(relation_value)
    if unsupported is not None:
        status, warning = unsupported
        z = p_value = float("nan")
        preferred = None
    elif not variance_positive or not math.isfinite(raw_z) or not math.isfinite(raw_p):
        status = ComparisonStatus.VARIANCE_DEGENERATE
        warning = (
            "Casewise log-likelihood differences have zero or numerically "
            "degenerate variance; preference inference is undefined."
        )
        z = p_value = float("nan")
        preferred = None
    else:
        z = raw_z
        p_value = raw_p
        warning = None
        if p_value >= alpha or z == 0.0:
            status = ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE
            preferred = None
        elif z > 0.0:
            status = ComparisonStatus.MODEL_A_PREFERRED
            preferred = label_a
        else:
            status = ComparisonStatus.MODEL_B_PREFERRED
            preferred = label_b

    return ModelComparisonResult(
        model_a=label_a,
        model_b=label_b,
        relation=relation_value,
        status=status,
        bic_correction=bic_correction,
        n_cases=len(values_a),
        k_a=int(k_a),
        k_b=int(k_b),
        raw_mean_loglik_difference=raw_mean,
        omega=omega,
        variance_positive=variance_positive,
        z=z,
        p_two_sided=p_value,
        preferred_model=preferred,
        warning=warning,
    )
