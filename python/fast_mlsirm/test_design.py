from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


def item_information(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    theta: np.ndarray | None = None,
    person_index: int | None = None,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Dichotomous item information for the simple-structure trait dimension.

    Numerical Fisher information is owned by the compiled Rust core
    (``cat_item_information``); Python validates shapes and marshals bank
    parameters. The latent-space position follows
    :func:`_person_params` (person ``xi`` when indexed, else the bank mean).
    """
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != params.alpha.shape:
        raise ValueError("factor_id length must match number of items")
    sub = _person_params(params, theta=theta, person_index=person_index)
    from . import _core as core

    n_dims = int(np.asarray(params.theta).shape[1])
    latent_dim = int(np.asarray(params.zeta).shape[1])
    result = core.cat_item_information(
        theta=np.ascontiguousarray(sub.theta.reshape(-1), dtype=np.float64),
        xi_mean=np.ascontiguousarray(sub.xi.reshape(-1), dtype=np.float64),
        alpha=np.ascontiguousarray(params.alpha, dtype=np.float64),
        b=np.ascontiguousarray(params.b, dtype=np.float64),
        zeta=np.ascontiguousarray(params.zeta, dtype=np.float64).reshape(-1),
        tau=float(params.tau),
        factor_id=np.ascontiguousarray(factors, dtype=np.int64),
        model=model,
        n_dims=n_dims,
        latent_dim=latent_dim,
        eps_distance=1e-8,
        device="auto",
    )
    return np.asarray(result, dtype=np.float64)


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
    item information at the given ``theta`` (or person). Ranking and exclusion
    are owned by the compiled Rust core (``cat_select_item``); Python validates
    and marshals controls.
    """
    factors = np.asarray(factor_id, dtype=np.int64)
    if factors.shape != params.alpha.shape:
        raise ValueError("factor_id length must match number of items")
    sub = _person_params(params, theta=theta, person_index=person_index)
    adm = None
    if administered is not None:
        used = np.asarray(administered, dtype=np.int64)
        n_items = int(np.asarray(params.b).shape[0])
        if used.size and (np.any(used < 0) or np.any(used >= n_items)):
            raise ValueError("administered item index out of range")
        adm = np.ascontiguousarray(np.unique(used), dtype=np.int64)

    from . import _core as core

    n_dims = int(np.asarray(params.theta).shape[1])
    latent_dim = int(np.asarray(params.zeta).shape[1])
    selected = core.cat_select_item(
        theta=np.ascontiguousarray(sub.theta.reshape(-1), dtype=np.float64),
        xi_mean=np.ascontiguousarray(sub.xi.reshape(-1), dtype=np.float64),
        administered=adm,
        alpha=np.ascontiguousarray(params.alpha, dtype=np.float64),
        b=np.ascontiguousarray(params.b, dtype=np.float64),
        zeta=np.ascontiguousarray(params.zeta, dtype=np.float64).reshape(-1),
        tau=float(params.tau),
        factor_id=np.ascontiguousarray(factors, dtype=np.int64),
        model=model,
        n_dims=n_dims,
        latent_dim=latent_dim,
        eps_distance=1e-8,
        device="auto",
    )
    return int(selected)


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

    Ordering, exclusion, and content-feasibility decisions are owned by the
    compiled Rust core (``assemble_test_form_greedy``); Python validates public
    shapes and marshals constraint maps without mutating caller arrays.
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

    exclude_list: list[int] = []
    if exclude is not None:
        exclude_list = [int(i) for i in np.asarray(exclude, dtype=np.int64).tolist()]

    from . import _core as core

    selected = core.assemble_test_form_greedy(
        np.ascontiguousarray(scores, dtype=np.float64),
        int(length),
        None if labels is None else [str(x) for x in labels.tolist()],
        min_counts,
        max_counts,
        exclude_list,
    )
    return np.asarray(selected, dtype=np.int64)


def _person_params(params: MLSIRMParams, theta: np.ndarray | None, person_index: int | None) -> MLSIRMParams:
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
        xi_row = params.xi[[person_index]] if person_index is not None else params.xi.mean(axis=0, keepdims=True)
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


