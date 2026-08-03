"""Exploratory factor rotation backed entirely by the Rust numerical core.

The API deliberately distinguishes a *rotation criterion* from an *observed
multi-start solution*.  Rotation objectives are usually non-convex, so
``rotate_factor_loadings`` reports basin support and every start-level objective
instead of claiming that the best finite-start result is a proven global
minimum.

References
----------
Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and
software for arbitrary rotation criteria in factor analysis. *Educational and
Psychological Measurement, 65*(5), 676–696.
https://doi.org/10.1177/0013164404272507

Browne, M. W. (2001). An overview of analytic rotation in exploratory factor
analysis. *Multivariate Behavioral Research, 36*(1), 111–150.
https://doi.org/10.1207/S15327906MBR3601_05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


@dataclass(frozen=True)
class RotationCriterionInfo:
    """Capability metadata for one built-in criterion identifier."""

    name: str
    family: str
    orthogonal: bool
    oblique: bool
    requires_target: bool
    description: str


@dataclass(frozen=True)
class RotationSolution:
    """Auditable best-observed solution from deterministic multi-start GPA.

    ``pattern_matrix`` contains regression-pattern loadings. For oblique
    rotation, ``structure_matrix`` equals ``pattern_matrix @
    factor_correlation``. ``basin_support`` counts starts whose objective lies
    within the configured relative tolerance of the selected value, while
    ``distinct_minima`` counts the observed objective basins.
    """

    pattern_matrix: np.ndarray
    structure_matrix: np.ndarray
    factor_correlation: np.ndarray
    transform_matrix: np.ndarray
    criterion: str
    mode: str
    criterion_value: float
    gradient_norm: float
    iterations: int
    converged: bool
    termination_reason: str
    best_start_index: int
    n_starts: int
    converged_starts: int
    basin_support: int
    distinct_minima: int
    start_values: np.ndarray
    start_converged: np.ndarray
    max_factor_correlation: float
    normalized: bool
    worker_count: int
    backend: str


_CATALOGUE: Final[tuple[RotationCriterionInfo, ...]] = (
    RotationCriterionInfo("quartimax", "orthomax", True, False, False, "Orthogonal variable-complexity minimization."),
    RotationCriterionInfo("varimax", "orthomax", True, False, False, "Orthogonal factor-variance maximization."),
    RotationCriterionInfo("orthomax", "orthomax", True, False, False, "Continuous Orthomax gamma family."),
    RotationCriterionInfo("crawford_ferguson", "crawford_ferguson", True, True, False, "Continuous Crawford–Ferguson kappa family."),
    RotationCriterionInfo("equamax", "crawford_ferguson", True, True, False, "Crawford–Ferguson equamax special case."),
    RotationCriterionInfo("parsimax", "crawford_ferguson", True, True, False, "Crawford–Ferguson parsimax special case."),
    RotationCriterionInfo("factor_parsimony", "crawford_ferguson", True, True, False, "Crawford–Ferguson factor-parsimony endpoint."),
    RotationCriterionInfo("oblimin", "oblimin", False, True, False, "Continuous direct-oblimin gamma family."),
    RotationCriterionInfo("quartimin", "oblimin", False, True, False, "Direct-oblimin quartimin special case."),
    RotationCriterionInfo("biquartimin", "oblimin", False, True, False, "Direct-oblimin gamma=.5 special case."),
    RotationCriterionInfo("covarimin", "oblimin", False, True, False, "Direct-oblimin gamma=1 special case."),
    RotationCriterionInfo("geomin", "geomin", True, True, False, "Geometric-mean row-complexity criterion."),
    RotationCriterionInfo("target", "target", True, True, True, "Complete or NaN-partially specified target rotation."),
    RotationCriterionInfo("pst", "target", True, True, True, "Weighted partially specified target rotation."),
    RotationCriterionInfo("entropy", "information", True, True, False, "Minimum entropy criterion."),
    RotationCriterionInfo("infomax", "information", True, True, False, "Infomax information criterion."),
    RotationCriterionInfo("mccammon", "information", True, False, False, "McCammon minimum entropy-ratio criterion."),
    RotationCriterionInfo("simplimax", "component_loss", False, True, False, "Kiers simplimax component-loss criterion."),
    RotationCriterionInfo("bifactor", "bifactor", True, True, False, "Jennrich–Bentler biquartimin criterion."),
    RotationCriterionInfo("bigeomin", "bifactor", True, True, False, "Jennrich–Bentler bi-geomin criterion."),
    RotationCriterionInfo("tandem_i", "tandem", True, False, False, "Comrey tandem criterion I."),
    RotationCriterionInfo("tandem_ii", "tandem", True, False, False, "Comrey tandem criterion II."),
    RotationCriterionInfo("oblimax", "oblimax", False, True, False, "Scale-invariant oblimax criterion."),
    RotationCriterionInfo("bentler", "invariant_simplicity", True, True, False, "Bentler invariant pattern-simplicity criterion."),
    RotationCriterionInfo("varimin", "anti_simple_structure", True, False, False, "Orthogonal varimin complement of varimax."),
    RotationCriterionInfo("lp_wls", "component_loss", True, True, False, "Weighted L2 kernel for iterative Lp/FSS rotation."),
)

_ORTHOGONAL_ONLY: Final[frozenset[str]] = frozenset(
    {"quartimax", "varimax", "varimin", "orthomax", "mccammon", "tandem_i", "tandem_ii"}
)
_OBLIQUE_ONLY: Final[frozenset[str]] = frozenset(
    {"oblimin", "quartimin", "biquartimin", "covarimin", "simplimax", "oblimax"}
)


def available_rotation_criteria() -> tuple[RotationCriterionInfo, ...]:
    """Return immutable metadata for every built-in criterion identifier."""

    return _CATALOGUE


def _method_name(value: str) -> str:
    """Normalize a public criterion identifier without silently guessing."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("criterion must be a non-empty string")
    return value.strip().lower().replace("-", "_")


def _mode_name(method: str, value: str | None) -> str:
    """Resolve an explicit mode or the criterion's unambiguous default."""

    if value is None:
        return "orthogonal" if method in _ORTHOGONAL_ONLY else "oblique"
    normalized = str(value).strip().lower()
    aliases = {"orth": "orthogonal", "t": "orthogonal", "oblq": "oblique", "q": "oblique"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"orthogonal", "oblique"}:
        raise ValueError("mode must be 'orthogonal' or 'oblique'")
    if method in _ORTHOGONAL_ONLY and normalized != "orthogonal":
        raise ValueError(f"{method} supports orthogonal rotation only")
    if method in _OBLIQUE_ONLY and normalized != "oblique":
        raise ValueError(f"{method} supports oblique rotation only")
    return normalized


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite contiguous two-dimensional float64 matrix."""

    matrix = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _optional_matrix(
    value: np.ndarray | None,
    name: str,
    shape: tuple[int, int],
    *,
    allow_nan: bool = False,
) -> np.ndarray | None:
    """Validate an optional target or weight matrix."""

    if value is None:
        return None
    matrix = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if allow_nan:
        if np.any(np.isinf(matrix)):
            raise ValueError(f"{name} may contain finite values or NaN, not infinity")
    elif not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _solution_from_core(result: tuple[dict, dict, dict, dict, dict]) -> RotationSolution:
    """Convert the Rust core's typed maps into a public immutable result."""

    arrays, floats, integers, booleans, strings = result
    rows = int(integers["n_rows"])
    factors = int(integers["n_factors"])
    return RotationSolution(
        pattern_matrix=np.asarray(arrays["pattern_matrix"], dtype=np.float64).reshape(rows, factors),
        structure_matrix=np.asarray(arrays["structure_matrix"], dtype=np.float64).reshape(rows, factors),
        factor_correlation=np.asarray(arrays["factor_correlation"], dtype=np.float64).reshape(factors, factors),
        transform_matrix=np.asarray(arrays["transform_matrix"], dtype=np.float64).reshape(factors, factors),
        criterion=strings["criterion"],
        mode=strings["mode"],
        criterion_value=float(floats["criterion_value"]),
        gradient_norm=float(floats["gradient_norm"]),
        iterations=int(integers["iterations"]),
        converged=bool(booleans["converged"]),
        termination_reason=strings["termination_reason"],
        best_start_index=int(integers["best_start_index"]),
        n_starts=int(integers["n_starts"]),
        converged_starts=int(integers["converged_starts"]),
        basin_support=int(integers["basin_support"]),
        distinct_minima=int(integers["distinct_minima"]),
        start_values=np.asarray(arrays["start_values"], dtype=np.float64),
        start_converged=np.asarray(arrays["start_converged"], dtype=np.float64).astype(bool),
        max_factor_correlation=float(floats["max_factor_correlation"]),
        normalized=bool(booleans["normalized"]),
        worker_count=int(integers["worker_count"]),
        backend=strings["backend"],
    )


def rotate_factor_loadings(
    loadings: np.ndarray,
    criterion: str = "geomin",
    *,
    mode: str | None = None,
    normalize: bool = False,
    n_starts: int = 32,
    seed: int = 1,
    max_iter: int = 2_000,
    tolerance: float = 1e-5,
    function_window: int = 10,
    max_line_search: int = 20,
    basin_tolerance: float = 1e-8,
    max_threads: int = 0,
    kappa: float | None = None,
    gamma: float | None = None,
    delta: float | None = None,
    simplimax_zeros: int | None = None,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> RotationSolution:
    """Rotate an unrotated loading matrix with deterministic multi-start GPA.

    Parameters
    ----------
    loadings:
        Finite ``(variables, factors)`` unrotated loading matrix.
    criterion:
        A name from :func:`available_rotation_criteria`.
    mode:
        ``"orthogonal"`` or ``"oblique"``. When omitted, criteria with only
        one supported manifold use it; criteria supporting both default to
        oblique rotation.
    normalize:
        Apply and later undo Kaiser row normalization.
    n_starts:
        Total deterministic starts including identity. More starts reduce, but
        never eliminate, local-minimum risk.
    target, weights:
        Target may contain ``NaN`` for unspecified cells. ``pst`` requires both
        target and non-negative weights.

    Returns
    -------
    RotationSolution
        The best observed solution and local-minimum diagnostics.
    """

    from . import _core

    matrix = _matrix(loadings, "loadings")
    rows, factors = map(int, matrix.shape)
    method = _method_name(criterion)
    resolved_mode = _mode_name(method, mode)
    target_matrix = _optional_matrix(target, "target", matrix.shape, allow_nan=True)
    weight_matrix = _optional_matrix(weights, "weights", matrix.shape)
    if weight_matrix is not None and np.any(weight_matrix < 0.0):
        raise ValueError("weights must be non-negative")
    result = _core.rotate_factor_loadings(
        matrix,
        method,
        resolved_mode,
        bool(normalize),
        int(n_starts),
        int(seed),
        int(max_iter),
        float(tolerance),
        int(function_window),
        int(max_line_search),
        float(basin_tolerance),
        int(max_threads),
        kappa,
        gamma,
        delta,
        simplimax_zeros,
        target_matrix,
        weight_matrix,
    )
    return _solution_from_core(result)


def rotation_criterion_value_gradient(
    loadings: np.ndarray,
    criterion: str,
    *,
    kappa: float | None = None,
    gamma: float | None = None,
    delta: float | None = None,
    simplimax_zeros: int | None = None,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return a built-in criterion value and analytic loading-space gradient.

    This diagnostic endpoint is useful for parity tests and criterion research;
    optimization remains in :func:`rotate_factor_loadings`.
    """

    from . import _core

    matrix = _matrix(loadings, "loadings")
    method = _method_name(criterion)
    target_matrix = _optional_matrix(target, "target", matrix.shape, allow_nan=True)
    weight_matrix = _optional_matrix(weights, "weights", matrix.shape)
    if weight_matrix is not None and np.any(weight_matrix < 0.0):
        raise ValueError("weights must be non-negative")
    value, gradient = _core.rotation_criterion_value_gradient(
        matrix,
        method,
        kappa,
        gamma,
        delta,
        simplimax_zeros,
        target_matrix,
        weight_matrix,
    )
    return float(value), np.asarray(gradient, dtype=np.float64).reshape(matrix.shape)
