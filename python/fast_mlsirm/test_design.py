from __future__ import annotations

import numpy as np

from .types import MLSIRMParams

_ITEM_INFORMATION_EPS_DISTANCE = 1e-8


def item_information(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray | None = None,
    person_index: int | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Dichotomous item information from the compiled Rust scoring core.

    Python validates the public simple-structure request, selects the requested
    person/trait point, and marshals immutable arrays. The probability and
    Fisher-information arithmetic is owned by ``mlsirm-core`` through the same
    bank-information entrypoint used by serving.
    """
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != params.alpha.shape:
        raise ValueError("factor_id length must match number of items")
    sub = _person_params(params, theta=theta, person_index=person_index)

    from . import _core as core

    theta_row = np.ascontiguousarray(sub.theta, dtype=np.float64).reshape(-1)
    xi_row = np.ascontiguousarray(sub.xi, dtype=np.float64).reshape(-1)
    alpha = np.ascontiguousarray(params.alpha, dtype=np.float64)
    intercept = np.ascontiguousarray(params.b, dtype=np.float64)
    zeta = np.ascontiguousarray(params.zeta, dtype=np.float64)
    factor_values = np.ascontiguousarray(factors, dtype=np.int64)
    result = dict(
        core.bank_information(
            theta_row,
            xi_row,
            1,
            alpha=alpha,
            b=intercept,
            zeta=zeta.reshape(-1),
            tau=float(params.tau),
            factor_id=factor_values,
            model=model,
            n_dims=int(sub.theta.shape[1]),
            latent_dim=int(sub.xi.shape[1]),
            eps_distance=_ITEM_INFORMATION_EPS_DISTANCE,
            device="auto",
        )
    )
    information = np.asarray(result["item_info"], dtype=np.float64)
    if information.shape != params.alpha.shape:
        raise RuntimeError("compiled Rust core returned an invalid item-information shape")
    return information


def select_cat_item(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray | None = None,
    person_index: int | None = None,
    administered: np.ndarray | None = None,
    model: str = "MLS2PLM",
) -> int:
    """Select the next CAT item by maximum Fisher information.

    Returns the index of the not-yet-``administered`` item with the highest
    item information at the given ``theta`` (or person), for adaptive testing.
    Item-information arithmetic is Rust-owned; the final global maximum policy
    remains Python-owned until the dedicated #629 selection slice lands.
    """
    information = item_information(
        params,
        factor_id,
        theta=theta,
        person_index=person_index,
        model=model,
    )
    candidates = information.copy()
    if administered is not None:
        used = np.asarray(administered, dtype=np.int64)
        if np.any(used < 0) or np.any(used >= candidates.size):
            raise ValueError("administered item index out of range")
        candidates[used] = -np.inf
    if not np.any(np.isfinite(candidates)):
        raise ValueError("no candidate items remain")
    return int(np.argmax(candidates))


def assemble_test_form(
    information: np.ndarray,
    length: int,
    content: np.ndarray | None = None,
    min_per_content: dict[str, int] | None = None,
    max_per_content: dict[str, int] | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a fixed-length test form by greedy maximum-information selection.

    Picks the ``length`` highest-information items (skipping ``exclude``d ones)
    subject to per-content min/max count constraints, raising if no feasible
    form satisfies them. Returns the selected item indices.
    """
    scores = np.asarray(information, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("information must be a 1D array")
    if length < 1 or length > scores.size:
        raise ValueError("length must be between 1 and the number of items")

    min_counts = {str(k): int(v) for k, v in (min_per_content or {}).items()}
    max_counts = {str(k): int(v) for k, v in (max_per_content or {}).items()}
    labels = None if content is None else np.asarray(content).astype(str)
    if (min_counts or max_counts) and labels is None:
        raise ValueError("content labels are required for content constraints")
    if labels is not None and labels.shape != scores.shape:
        raise ValueError("content length must match information")

    excluded = (
        set(np.asarray(exclude, dtype=np.int64).tolist())
        if exclude is not None
        else set()
    )
    selected: list[int] = []
    counts: dict[str, int] = {}
    order = [
        int(i)
        for i in np.argsort(-scores)
        if i not in excluded and np.isfinite(scores[i])
    ]

    for _ in range(length):
        for item in order:
            if item in selected:
                continue
            label = None if labels is None else str(labels[item])
            next_counts = counts.copy()
            if label is not None:
                if next_counts.get(label, 0) >= max_counts.get(label, length):
                    continue
                next_counts[label] = next_counts.get(label, 0) + 1
            if _constraints_feasible(
                order,
                selected + [item],
                excluded,
                labels,
                next_counts,
                length,
                min_counts,
                max_counts,
            ):
                selected.append(item)
                counts = next_counts
                break
        else:
            raise ValueError("could not assemble a form that satisfies constraints")

    for label, minimum in min_counts.items():
        if counts.get(label, 0) < minimum:
            # Unreachable: the per-slot feasibility look-ahead only admits a pick
            # when every minimum can still be met, so no completed length-form can
            # leave a minimum unsatisfied here. Kept as a defensive guard.
            raise ValueError(
                f"minimum content constraint not met: {label}"
            )  # pragma: no cover
    return np.asarray(selected, dtype=np.int64)


def _person_params(
    params: MLSIRMParams,
    theta: np.ndarray | None,
    person_index: int | None,
) -> MLSIRMParams:
    """Return a single-person parameter view for item-information evaluation.

    When ``theta`` is given it is used directly, and the latent-space position
    ``xi`` is that person's row when ``person_index`` is supplied, otherwise the
    mean ``xi`` across all persons. When ``theta`` is ``None`` both ``theta`` and
    ``xi`` are taken from ``person_index``, defaulting to the first person.
    """
    if theta is None:
        if person_index is None:
            person_index = 0
        theta_row = params.theta[[person_index]]
        xi_row = params.xi[[person_index]]
    else:
        theta_row = np.asarray(theta, dtype=np.float64).reshape(1, -1)
        xi_row = (
            params.xi[[person_index]]
            if person_index is not None
            else params.xi.mean(axis=0, keepdims=True)
        )
    if theta_row.shape[1] != params.theta.shape[1]:
        raise ValueError("theta dimensionality must match params")
    return MLSIRMParams(
        theta=theta_row,
        alpha=params.alpha,
        b=params.b,
        xi=xi_row,
        zeta=params.zeta,
        tau=params.tau,
    )


def _constraints_feasible(
    order: list[int],
    selected: list[int],
    excluded: set[int],
    labels: np.ndarray | None,
    counts: dict[str, int],
    length: int,
    min_counts: dict[str, int],
    max_counts: dict[str, int],
) -> bool:
    """Return whether the remaining slots can still satisfy the min-content constraints.

    A look-ahead feasibility check used during greedy form assembly: verifies
    enough eligible items remain per content area to meet each minimum.
    """
    slots_left = length - len(selected)
    required_left = sum(
        max(0, minimum - counts.get(label, 0))
        for label, minimum in min_counts.items()
    )
    if required_left > slots_left:
        return False
    if labels is None:
        return True

    blocked = set(selected) | excluded
    for label, minimum in min_counts.items():
        needed = max(0, minimum - counts.get(label, 0))
        available = 0
        for item in order:
            if item in blocked or str(labels[item]) != label:
                continue
            if counts.get(label, 0) + available >= max_counts.get(label, length):
                break
            available += 1
        if available < needed:
            return False
    return True
