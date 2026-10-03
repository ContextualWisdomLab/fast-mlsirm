"""Reference-metric person scoring for fitted two-tier GRM item banks.

This module owns scoring, not calibration. The caller supplies a fitted
orthogonal-primary two-tier fit (normally :class:`TwoTierGrmFit` with
``primary_identification == "orthogonal"``); its Rust-computed primary EAPs
are reused as-is and each person's expected raw total comes from
:func:`expected_raw_two_tier_grm`, the single owner of the two-tier expected
raw score. Anchor rows identify the reference item metric (Kim, 2006), and
the reference expected-score moments integrate the same expected raw total
(Lord, 1980) over the reference primary prior using anchor rows only. Each
specific factor is N(0, 1) and integrated one block at a time (Cai, 2010;
Gibbons et al., 2007). A correlated ``Phi`` needs a different primary
integral and fails closed.

All numerical work runs in the Rust core (ADR-0002); Python validates inputs
and marshals arrays. The module is domain-neutral: no instrument labels.

References (APA 7th ed.)

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a
microcomputer environment. *Applied Psychological Measurement, 6*(4),
431-444. https://doi.org/10.1177/014662168200600405

Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. *Psychometrika, 75*(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0

Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. *Psychometrika, 80*(2), 535-559.
https://doi.org/10.1007/s11336-014-9411-3

Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
Full-information item bifactor analysis of graded response data. *Applied
Psychological Measurement, 31*(1), 4-19.
https://doi.org/10.1177/0146621606289485

Kim, S. (2006). A comparative study of IRT fixed parameter calibration
methods. *Journal of Educational Measurement, 43*(4), 355-381.
https://doi.org/10.1111/j.1745-3984.2006.00021.x

Lord, F. M. (1980). *Applications of item response theory to practical
testing problems*. Lawrence Erlbaum Associates.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .polytomous import MAX_POLY_QUADRATURE_POINTS
from .two_tier_grm import _specific_map_control, expected_raw_two_tier_grm


def _integer(value: object, name: str) -> int:
    """Validate a Python integer control without silently coercing floats."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    if value > MAX_POLY_QUADRATURE_POINTS:
        raise ValueError(f"{name} must be <= {MAX_POLY_QUADRATURE_POINTS}")
    return value


def _vector(value: object, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or accept an exact length-``n`` vector."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},), got {arr.shape}")
    return np.ascontiguousarray(arr)


def _check_orthogonal(fit) -> None:
    required = (
        "a_primary", "a_specific", "threshold", "phi", "n_primary", "n_specific",
        "n_cat", "theta_p_eap", "theta_p_sd", "primary_identification",
    )
    missing = [name for name in required if not hasattr(fit, name)]
    if missing:
        raise TypeError(f"fit is missing required attributes: {missing}")
    if fit.primary_identification != "orthogonal":
        raise ValueError(
            "two-tier reference scoring requires primary_identification == "
            "'orthogonal' (Phi fixed to I during estimation)"
        )
    n_primary = int(fit.n_primary)
    if not np.array_equal(np.asarray(fit.phi, dtype=np.float64), np.eye(n_primary)):
        raise ValueError("two-tier reference scoring requires Phi == I")
    threshold = np.asarray(fit.threshold, dtype=np.float64)
    if threshold.ndim != 2 or not np.all(np.isfinite(threshold)):
        raise ValueError("fit.threshold must be a finite n_items x (n_cat - 1) array")
    if np.any(np.diff(threshold, axis=1) >= 0.0):
        raise ValueError("fit.threshold rows must be strictly decreasing")


@dataclass(frozen=True)
class TwoTierReferenceExpectedScoreMoments:
    """Moments of the anchor-only expected raw score under the reference prior."""

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
    """Person scores on the reference item metric."""

    theta_primary_eap: np.ndarray
    theta_primary_sd: np.ndarray
    expected_raw: np.ndarray
    reference_moments: TwoTierReferenceExpectedScoreMoments
    fit: object


def two_tier_reference_expected_score_moments(
    fit,
    specific_map: np.ndarray,
    anchor: np.ndarray,
    *,
    primary_mean: object = 0.0,
    primary_sd: object = 1.0,
    q_primary: int,
    q_specific: int,
) -> TwoTierReferenceExpectedScoreMoments:
    """Integrate the anchor-only expected raw total over the reference prior."""
    q_p = _integer(q_primary, "q_primary")
    q_s = _integer(q_specific, "q_specific")
    _check_orthogonal(fit)
    n_primary, n_specific, n_cat = int(fit.n_primary), int(fit.n_specific), int(fit.n_cat)
    a_specific = np.asarray(fit.a_specific, dtype=np.float64)
    n_items = a_specific.shape[0]
    smap = _specific_map_control(specific_map, n_items, n_specific)
    mask = np.asarray(anchor)
    if mask.dtype.kind != "b" or mask.shape != (n_items,) or not mask.any():
        raise ValueError("anchor must be a boolean mask of shape (n_items,) with at least one item")
    a_primary = np.asarray(fit.a_primary, dtype=np.float64)
    threshold = np.asarray(fit.threshold, dtype=np.float64)
    if a_primary.shape != (n_items, n_primary) or threshold.shape != (n_items, n_cat - 1):
        raise ValueError("fit.a_primary/threshold shapes are inconsistent with n_items")
    p_mean = _vector(primary_mean, n_primary, "primary_mean")
    p_sd = _vector(primary_sd, n_primary, "primary_sd")
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "two_tier_grm_reference_score_moments"):
        raise RuntimeError("two-tier reference scoring requires the compiled Rust core")
    out = core.two_tier_grm_reference_score_moments(
        np.ascontiguousarray(a_primary[mask]).reshape(-1),
        np.ascontiguousarray(a_specific[mask]),
        np.ascontiguousarray(threshold[mask]).reshape(-1),
        np.ascontiguousarray(smap[mask]),
        n_cat,
        n_primary,
        n_specific,
        p_mean,
        p_sd,
        q_p,
        q_s,
    )
    return TwoTierReferenceExpectedScoreMoments(
        float(out["mean"]), float(out["second_moment"]), float(out["variance"]),
        p_mean, p_sd, int(mask.sum()), q_p, q_s,
    )


def score_two_tier_fipc_group_persons(
    fit,
    specific_map: np.ndarray,
    anchor: np.ndarray,
    *,
    q_primary: int,
    q_specific: int,
    reference_primary_mean: object = 0.0,
    reference_primary_sd: object = 1.0,
) -> TwoTierFipcGroupPersonScores:
    """Score the fit's persons on the reference item metric.

    ``theta_primary_eap``/``theta_primary_sd`` are the fit's Rust-computed
    EAPs (Bock & Mislevy, 1982); no latent rescaling is performed.
    ``expected_raw`` is :func:`expected_raw_two_tier_grm` at those EAPs.
    Reference moments use anchor rows and the reference primary prior only.
    """
    reference = two_tier_reference_expected_score_moments(
        fit,
        specific_map,
        anchor,
        primary_mean=reference_primary_mean,
        primary_sd=reference_primary_sd,
        q_primary=q_primary,
        q_specific=q_specific,
    )
    expected_raw = expected_raw_two_tier_grm(fit, specific_map, reference.q_specific)
    eap = np.asarray(fit.theta_p_eap, dtype=np.float64)
    return TwoTierFipcGroupPersonScores(
        eap, np.asarray(fit.theta_p_sd, dtype=np.float64), expected_raw, reference, fit
    )


def execute_two_tier_fipc_group_person_score_payload(payload: dict[str, object]) -> dict[str, object]:
    """Execute a JSON-serializable scorer payload whose ``fit`` is a field mapping."""
    if type(payload) is not dict:
        raise TypeError("payload must be a dict")
    required = ("fit", "specific_map", "anchor", "q_primary", "q_specific")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"payload missing required keys: {missing}")
    if type(payload["fit"]) is not dict:
        raise TypeError("payload['fit'] must be a dict of fit fields")
    optional = ("reference_primary_mean", "reference_primary_sd")
    result = score_two_tier_fipc_group_persons(
        SimpleNamespace(**payload["fit"]),
        np.asarray(payload["specific_map"]),
        np.asarray(payload["anchor"]),
        q_primary=payload["q_primary"],
        q_specific=payload["q_specific"],
        **{key: payload[key] for key in optional if key in payload},
    )
    return {
        "family": "two_tier_fipc_group_person_score",
        "model_scope": "two_tier_grm_orthogonal_primary",
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
