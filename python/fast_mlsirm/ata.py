"""Automated test assembly (ATA) toward a target information function.

This module is a *downstream application* on already-calibrated item
parameters. It does not touch the calibration objective, likelihood, gradients,
or the MLS2PLM formula; it reuses the repository's 2PL Fisher item information
(:func:`fast_mlsirm.test_design.item_information`) to assemble a fixed-length
form whose test information function (TIF) approximates a caller-specified
target ``T(theta_k)`` at a set of trait points, subject to length, content, and
basic exposure constraints.

Relationship to :func:`fast_mlsirm.test_design.assemble_test_form`
------------------------------------------------------------------
``assemble_test_form`` maximizes a single precomputed item-information vector
under content constraints. This module instead targets a *test information
function evaluated at several trait points* and adds exposure control, matching
the optimal-test-design framing of van der Linden (2005): the assembled form's
TIF is driven toward the target across all specified ``theta`` points rather
than maximized at one point.

Method
------
The exact formulation of van der Linden (2005) is a 0-1 integer program that
selects ``x_i in {0,1}`` to make ``I(theta_k) = sum_i x_i I_i(theta_k)`` meet a
target ``T(theta_k)`` subject to ``sum_i x_i = n`` and content/exposure/enemy
constraints. To stay dependency-free (no SciPy/solver), this uses the standard
greedy surrogate: at each step add the eligible item that most reduces the total
*capped shortfall* ``sum_k max(0, T_k - I(theta_k))`` -- i.e. maximizes
``sum_k [min(T_k, I_k + I_ik) - min(T_k, I_k)]``. Capping at the target rewards
covering under-informed points and does not reward overshoot, so information is
spread to match the target curve. This is deterministic given ``seed`` (used
only to break exact ties reproducibly).

References (APA 7th ed.)
------------------------
van der Linden, W. J. (2005). *Linear models for optimal test design*.
    Springer. https://doi.org/10.1007/0-387-29054-0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .test_design import item_information
from .types import MLSIRMParams

__all__ = [
    "AssembledForm",
    "assemble_to_target",
    "item_information_matrix",
]

_TIE_EPS = 1e-12


@dataclass
class AssembledForm:
    """Result of a target-information assembly.

    Attributes
    ----------
    items:
        Selected item indices (assembly order).
    achieved_info:
        Achieved test information ``sum_i I_i(theta_k)`` at each target point.
    target_info:
        The requested target information at each target point.
    shortfall:
        Total capped shortfall ``sum_k max(0, T_k - achieved_k)`` (0 when the
        target is met at every point).
    content_counts:
        Count of selected items per content label (empty when no content was
        supplied).
    """

    items: np.ndarray
    achieved_info: np.ndarray
    target_info: np.ndarray
    shortfall: float
    content_counts: dict[str, int]


def _target_theta_rows(target_thetas: np.ndarray, n_dims: int) -> np.ndarray:
    """Coerce target trait points to a ``(n_points, n_dims)`` array."""
    thetas = np.asarray(target_thetas, dtype=np.float64)
    if thetas.ndim == 1:
        if n_dims == 1:
            thetas = thetas[:, None]
        elif thetas.shape == (n_dims,):
            thetas = thetas[None, :]
        else:
            raise ValueError("target_thetas must have shape (n_points, n_dims)")
    if thetas.ndim != 2 or thetas.shape[1] != n_dims:
        raise ValueError("target_thetas must have shape (n_points, n_dims)")
    if thetas.shape[0] < 1:
        raise ValueError("at least one target theta point is required")
    if not np.all(np.isfinite(thetas)):
        raise ValueError("target_thetas must be finite")
    return thetas


def item_information_matrix(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    target_thetas: np.ndarray,
    *,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Return the ``(n_points, n_items)`` item-information matrix.

    Entry ``[k, i]`` is the reused 2PL Fisher information ``I_i(theta_k)`` of
    item ``i`` at target trait point ``k`` (holding the latent-space position at
    the bank's population mean, as in
    :func:`fast_mlsirm.test_design.item_information`).
    """
    n_dims = int(np.asarray(bank.theta).shape[1])
    thetas = _target_theta_rows(target_thetas, n_dims)
    n_items = int(np.asarray(bank.b).shape[0])
    matrix = np.empty((thetas.shape[0], n_items), dtype=np.float64)
    for k in range(thetas.shape[0]):
        matrix[k] = item_information(bank, factor_id, theta=thetas[k], model=model)
    return matrix


def _content_feasible(
    labels: np.ndarray | None,
    selected: list[int],
    counts: dict[str, int],
    eligible_now: np.ndarray,
    length: int,
    min_counts: dict[str, int],
) -> bool:
    """Check that remaining min-content requirements can still be satisfied."""
    slots_left = length - len(selected)
    required_left = sum(max(0, m - counts.get(lbl, 0)) for lbl, m in min_counts.items())
    if required_left > slots_left:
        return False
    if labels is None or not min_counts:
        return True
    for lbl, minimum in min_counts.items():
        needed = max(0, minimum - counts.get(lbl, 0))
        if needed == 0:
            continue
        available = int(np.sum(labels[eligible_now] == lbl))
        if available < needed:
            return False
    return True


def assemble_to_target(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    target_thetas: np.ndarray,
    target_info: np.ndarray,
    length: int,
    *,
    model: str = "MLS2PLM",
    content: np.ndarray | None = None,
    min_per_content: dict[str, int] | None = None,
    max_per_content: dict[str, int] | None = None,
    exclude: np.ndarray | None = None,
    exposure_counts: dict[int, int] | None = None,
    exposure_max: int | None = None,
    seed: int = 0,
) -> AssembledForm:
    """Greedily assemble a length-``length`` form matching a target TIF.

    Selects items to drive ``sum_i I_i(theta_k)`` toward ``target_info[k]`` at
    every ``target_thetas`` point (van der Linden, 2005), honoring:

    * length (``length`` items),
    * content (``min_per_content`` / ``max_per_content`` keyed by ``content``
      labels),
    * exclusions (``exclude`` item indices), and
    * exposure: an item is ineligible when its prior usage count in
      ``exposure_counts`` reaches ``exposure_max``.

    Deterministic given ``seed``. Raises ``ValueError`` if no form satisfying
    the constraints can be assembled.
    """
    matrix = item_information_matrix(bank, factor_id, target_thetas, model=model)
    n_points, n_items = matrix.shape
    target = np.asarray(target_info, dtype=np.float64).ravel()
    if target.shape != (n_points,):
        raise ValueError("target_info must have one entry per target theta point")
    if np.any(target < 0) or not np.all(np.isfinite(target)):
        raise ValueError("target_info must be finite and non-negative")
    if not isinstance(length, (int, np.integer)) or isinstance(length, bool):
        raise ValueError("length must be an integer")
    length = int(length)
    if not (1 <= length <= n_items):
        raise ValueError("length must be between 1 and the number of items")

    labels = None if content is None else np.asarray(content).astype(str)
    min_counts = {str(k): int(v) for k, v in (min_per_content or {}).items()}
    max_counts = {str(k): int(v) for k, v in (max_per_content or {}).items()}
    if (min_counts or max_counts) and labels is None:
        raise ValueError("content labels are required for content constraints")
    if labels is not None and labels.shape != (n_items,):
        raise ValueError("content length must match the number of items")

    excluded = set(np.asarray(exclude, dtype=np.int64).tolist()) if exclude is not None else set()
    exposure_counts = {int(k): int(v) for k, v in (exposure_counts or {}).items()}
    if exposure_max is not None and exposure_max < 0:
        raise ValueError("exposure_max must be non-negative")

    raw_info = matrix.sum(axis=0)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    counts: dict[str, int] = {}
    accum = np.zeros(n_points, dtype=np.float64)

    for _ in range(length):
        # Eligibility mask independent of the current partial selection.
        eligible = np.ones(n_items, dtype=bool)
        for i in selected:
            eligible[i] = False
        for i in excluded:
            if 0 <= i < n_items:
                eligible[i] = False
        if exposure_max is not None:
            for i in range(n_items):
                if exposure_counts.get(i, 0) >= exposure_max:
                    eligible[i] = False
        if labels is not None:
            for i in np.nonzero(eligible)[0]:
                lbl = str(labels[i])
                if counts.get(lbl, 0) >= max_counts.get(lbl, length):
                    eligible[i] = False

        # Only consider items whose selection keeps the min-content requirements
        # satisfiable given the still-eligible pool after picking that item.
        candidates: list[int] = []
        for i in np.nonzero(eligible)[0]:
            remaining = eligible.copy()
            remaining[i] = False
            next_counts = dict(counts)
            if labels is not None:
                next_counts[str(labels[i])] = next_counts.get(str(labels[i]), 0) + 1
            if _content_feasible(labels, selected + [int(i)], next_counts, remaining, length, min_counts):
                candidates.append(int(i))
        if not candidates:
            raise ValueError("could not assemble a form that satisfies the constraints")

        cand = np.asarray(candidates, dtype=np.int64)
        gain = np.array(
            [float(np.sum(np.minimum(target, accum + matrix[:, i]) - np.minimum(target, accum))) for i in cand]
        )
        best_gain = float(np.max(gain))
        tied = cand[gain >= best_gain - _TIE_EPS]
        best_raw = float(np.max(raw_info[tied]))
        tied = tied[raw_info[tied] >= best_raw - _TIE_EPS]
        chosen = int(tied[0]) if tied.size == 1 else int(rng.choice(tied))

        selected.append(chosen)
        accum = accum + matrix[:, chosen]
        if labels is not None:
            counts[str(labels[chosen])] = counts.get(str(labels[chosen]), 0) + 1

    for lbl, minimum in min_counts.items():
        if counts.get(lbl, 0) < minimum:
            raise ValueError(f"minimum content constraint not met: {lbl}")

    shortfall = float(np.sum(np.maximum(0.0, target - accum)))
    return AssembledForm(
        items=np.asarray(selected, dtype=np.int64),
        achieved_info=accum,
        target_info=target,
        shortfall=shortfall,
        content_counts=dict(counts),
    )
