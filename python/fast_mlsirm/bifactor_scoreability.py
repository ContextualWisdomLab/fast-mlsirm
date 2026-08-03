"""Typed Python access to Rust-native bifactor scoreability diagnostics.

The functions in this module perform shape validation and memory-layout
marshalling only. ECV variants, item ECV, strict-pattern PUC, omega total,
omega hierarchical, and construct replicability ``H`` are computed in
``mlsirm_core::bifactor_indices``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._bifactor_core_loader import bifactor_core

MAX_BIFACTOR_ITEMS = 1_000_000
MAX_BIFACTOR_FACTORS = 64


@dataclass(frozen=True)
class BifactorScoreabilityResult:
    """Immutable scoreability diagnostics for one orthogonal bifactor solution.

    Attributes
    ----------
    factor_item_counts:
        Number of structurally active item loadings on each factor.
    is_strict_bifactor:
        Whether every item has the declared general loading and at most one
        active specific-factor loading.
    puc:
        Percentage of uncontaminated correlations for a strict bifactor
        pattern, otherwise ``None`` for cross-loaded specific structures.
    ecv_ss, ecv_sg, ecv_gs:
        Factor-level explained-common-variance variants.
    item_ecv:
        General-factor share of each item's common variance.
    omega_total, omega_hierarchical:
        Continuous latent-response reliability summaries for each factor's
        item domain. These are not categorical observed-score omega values.
    construct_replicability:
        Construct replicability coefficient ``H`` for each factor.
    """

    factor_item_counts: tuple[int, ...]
    is_strict_bifactor: bool
    puc: float | None
    ecv_ss: np.ndarray
    ecv_sg: np.ndarray
    ecv_gs: np.ndarray
    item_ecv: np.ndarray
    omega_total: np.ndarray
    omega_hierarchical: np.ndarray
    construct_replicability: np.ndarray


def _matrix(value: Any, name: str) -> np.ndarray:
    """Return a bounded contiguous finite-shape float64 matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D item-by-factor matrix")
    n_items, n_factors = map(int, matrix.shape)
    if not 2 <= n_items <= MAX_BIFACTOR_ITEMS:
        raise ValueError(
            f"{name} must contain between 2 and {MAX_BIFACTOR_ITEMS} items"
        )
    if not 2 <= n_factors <= MAX_BIFACTOR_FACTORS:
        raise ValueError(
            f"{name} must contain between 2 and {MAX_BIFACTOR_FACTORS} factors"
        )
    return np.ascontiguousarray(matrix)


def _uniqueness_vector(value: Any, n_items: int) -> np.ndarray:
    """Return a contiguous uniqueness vector with the required item count."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("uniquenesses must be a 1-D item vector")
    if vector.shape[0] != n_items:
        raise ValueError(
            f"uniquenesses length must equal the loading row count ({n_items})"
        )
    return np.ascontiguousarray(vector)


def _readonly_vector(value: Any, name: str) -> np.ndarray:
    """Return one immutable float64 result vector from the compiled mapping."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise RuntimeError(f"compiled bifactor result field {name} must be 1-D")
    vector.setflags(write=False)
    return vector


def _result_from_mapping(raw: dict[str, Any]) -> BifactorScoreabilityResult:
    """Build the typed immutable result without reproducing any formulas."""
    puc_value = raw["puc"]
    return BifactorScoreabilityResult(
        factor_item_counts=tuple(int(value) for value in raw["factor_item_counts"]),
        is_strict_bifactor=bool(raw["is_strict_bifactor"]),
        puc=None if puc_value is None else float(puc_value),
        ecv_ss=_readonly_vector(raw["ecv_ss"], "ecv_ss"),
        ecv_sg=_readonly_vector(raw["ecv_sg"], "ecv_sg"),
        ecv_gs=_readonly_vector(raw["ecv_gs"], "ecv_gs"),
        item_ecv=_readonly_vector(raw["item_ecv"], "item_ecv"),
        omega_total=_readonly_vector(raw["omega_total"], "omega_total"),
        omega_hierarchical=_readonly_vector(
            raw["omega_hierarchical"], "omega_hierarchical"
        ),
        construct_replicability=_readonly_vector(
            raw["construct_replicability"], "construct_replicability"
        ),
    )


def bifactor_scoreability(
    loadings: Any,
    uniquenesses: Any,
    *,
    general_factor: int = 0,
    zero_tolerance: float = 0.0,
) -> BifactorScoreabilityResult:
    """Compute scoreability indices from standardized orthogonal loadings.

    Parameters
    ----------
    loadings:
        Item-by-factor standardized loading matrix. Every item must load on the
        declared general factor.
    uniquenesses:
        Item residual variances. The Rust core requires each row to satisfy
        ``sum(loadings**2) + uniqueness = 1`` within ``1e-8``.
    general_factor:
        Zero-based general-factor column.
    zero_tolerance:
        Non-negative threshold used only to classify structural zeroes.
        Numerical sums retain the supplied loading values.

    Returns
    -------
    BifactorScoreabilityResult
        Immutable diagnostics computed entirely by the Rust core.

    Notes
    -----
    These indices support post-fit score interpretation; they do not select a
    bifactor model or define universal pass/fail cutoffs. Model selection,
    predictive validation, recovery, invariance, and substantive validity
    remain separate evidence requirements.
    """
    loading_matrix = _matrix(loadings, "loadings")
    uniqueness_vector = _uniqueness_vector(
        uniquenesses,
        int(loading_matrix.shape[0]),
    )
    raw = bifactor_core().bifactor_indices(
        loading_matrix,
        uniqueness_vector,
        general_factor,
        zero_tolerance,
    )
    return _result_from_mapping(raw)


def bifactor_scoreability_from_logit_slopes(
    logit_slopes: Any,
    *,
    general_factor: int = 0,
    zero_tolerance: float = 0.0,
) -> BifactorScoreabilityResult:
    """Compute latent-response indices from orthogonal logistic IRT slopes.

    The Rust core applies the logistic residual-variance convention
    ``pi**2 / 3`` and computes, for item ``i``,

    ``lambda_if = a_if / sqrt(sum_h(a_ih**2) + pi**2/3)``

    and

    ``psi_i = (pi**2/3) / (sum_h(a_ih**2) + pi**2/3)``.

    Scaling is performed in overflow-resistant row coordinates in Rust. The
    resulting omega coefficients describe the continuous latent-response
    representation, not observed binary or ordinal sum-score reliability.
    """
    slope_matrix = _matrix(logit_slopes, "logit_slopes")
    raw = bifactor_core().bifactor_indices_from_logit_slopes(
        slope_matrix,
        general_factor,
        zero_tolerance,
    )
    return _result_from_mapping(raw)
