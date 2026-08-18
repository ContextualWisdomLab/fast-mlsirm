"""Typed Python access to Rust-native multilevel contextual kernels.

This module performs marshalling only. The additive predictor
``sum_h w_ph * u_h`` and the MAP estimator of crossed / multiple-membership
person effects ``u_h`` are owned by ``mlsirm_core::multilevel``. See that
crate and the Fox & Glas (2001) / Browne, Goldstein, and Rasbash (2001)
citations on the public estimator.

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
IRT model. *Psychometrika, 66*, 271-288. https://doi.org/10.1007/BF02294839

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
103-124. https://doi.org/10.1177/1471082X0100100202
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .._multilevel_core_loader import multilevel_core
from ._validation import exact_integer
from .contracts import ContextMembershipDesign

ContextKey = tuple[str, str]


def _snapshot_context_effects(
    context_keys: tuple[ContextKey, ...],
    context_effects: Mapping[ContextKey, float],
) -> np.ndarray:
    """Read each required caller effect once without alien membership probes."""
    missing: list[ContextKey] = []
    values: list[float] = []
    for key in context_keys:
        try:
            value = context_effects[key]
        except KeyError:
            missing.append(key)
            continue
        except Exception:
            raise ValueError("context_effects could not be read safely") from None
        try:
            values.append(float(value))
        except Exception:
            raise ValueError(
                "context_effects values could not be converted safely"
            ) from None
    if missing:
        raise KeyError(f"context_effects is missing keys: {missing!r}")
    return np.array(values, dtype=np.float64)


def weighted_contextual_effect(
    design: ContextMembershipDesign,
    context_effects: Mapping[ContextKey, float],
    *,
    worker_count: int = 1,
) -> np.ndarray:
    """Return each observation's weighted contextual random-effect contribution.

    Parameters
    ----------
    design:
        A package-built ``ContextMembershipDesign``
        (``build_context_membership_design``). Its integrity is verified
        before use, so a tampered or hand-constructed design raises here
        rather than silently producing a wrong result.
    context_effects:
        Mapping from ``(context_dimension_id, context_id)`` to its current
        random-effect value ``u_h``. Must contain an entry for every context
        key ``design`` references. Required values are snapshotted exactly once
        without invoking caller-defined membership callbacks.
    worker_count:
        Number of deterministic worker threads (``>= 1``); the result does
        not depend on this value (see the Rust core's determinism proof).

    Returns
    -------
    numpy.ndarray
        One weighted contextual effect per observation, aligned with
        ``design.observation_ids``.

    Raises
    ------
    ValueError
        If ``worker_count < 1``, the design fails integrity verification, or a
        caller effect cannot be read/converted safely. Numerical finiteness is
        validated by the Rust core after marshalling.
    KeyError
        If ``context_effects`` is missing a key ``design`` references.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if type(design) is not ContextMembershipDesign:
        raise ValueError("design must be an exact ContextMembershipDesign")
    # Accessing the fingerprint triggers the design's own integrity
    # verification (raises on any post-factory tampering); the value itself
    # is not otherwise needed here.
    _ = design.design_fingerprint

    context_keys = design.context_keys
    key_index = {key: index for index, key in enumerate(context_keys)}
    effects = _snapshot_context_effects(context_keys, context_effects)

    by_observation: dict[str, list] = {
        observation_id: [] for observation_id in design.observation_ids
    }
    for edge in design.memberships:
        by_observation[edge.observation_id].append(edge)

    row_offsets: list[int] = [0]
    context_indices: list[int] = []
    weights: list[float] = []
    for observation_id in design.observation_ids:
        for edge in by_observation[observation_id]:
            context_indices.append(
                key_index[(edge.context_dimension_id, edge.context_id)]
            )
            weights.append(edge.membership_weight)
        row_offsets.append(len(context_indices))

    core = multilevel_core()
    return core.weighted_contextual_effect(
        np.array(row_offsets, dtype=np.uint64),
        np.array(context_indices, dtype=np.uint64),
        np.array(weights, dtype=np.float64),
        effects,
        worker_count,
    )


@dataclass(frozen=True)
class CrossedPersonEffectResult:
    """Immutable MAP estimate of crossed / multiple-membership ``u_h``.

    Attributes
    ----------
    context_effects:
        Mapping from ``(context_dimension_id, context_id)`` to the centered
        estimated random effect. Keys follow ``design.context_keys``.
    effect_vector:
        The same effects as a float64 vector aligned with ``context_keys``.
    context_keys:
        Deterministic dimension-qualified context identities.
    loglik:
        Bernoulli log-likelihood plus the Gaussian prior penalty.
    n_iter:
        Newton / IRLS iterations actually performed.
    converged:
        Whether the last effect step satisfied the requested tolerance.
    used_gpu:
        Whether the person-score reduction used the wgpu kernel.
    termination_reason:
        ``converged`` or ``max_iter_reached``.
    """

    context_effects: dict[ContextKey, float]
    effect_vector: np.ndarray
    context_keys: tuple[ContextKey, ...]
    loglik: float
    n_iter: int
    converged: bool
    used_gpu: bool
    termination_reason: str


def _csr_from_design(
    design: ContextMembershipDesign,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[ContextKey, ...]]:
    """Marshal one sealed design into CSR arrays and classification offsets."""
    context_keys = design.context_keys
    key_index = {key: index for index, key in enumerate(context_keys)}
    by_observation: dict[str, list] = {
        observation_id: [] for observation_id in design.observation_ids
    }
    for edge in design.memberships:
        by_observation[edge.observation_id].append(edge)

    row_offsets: list[int] = [0]
    context_indices: list[int] = []
    weights: list[float] = []
    for observation_id in design.observation_ids:
        for edge in by_observation[observation_id]:
            context_indices.append(
                key_index[(edge.context_dimension_id, edge.context_id)]
            )
            weights.append(edge.membership_weight)
        row_offsets.append(len(context_indices))

    classification_offsets = [0]
    for dimension_id in design.context_dimension_ids:
        classification_offsets.append(
            classification_offsets[-1]
            + sum(key[0] == dimension_id for key in context_keys)
        )
    return (
        np.array(row_offsets, dtype=np.uint64),
        np.array(context_indices, dtype=np.uint64),
        np.array(weights, dtype=np.float64),
        np.array(classification_offsets, dtype=np.uint64),
        context_keys,
    )


def _finite_vector(values: object, name: str, length: int) -> np.ndarray:
    """Return one exact 1-D float64 vector of the required length."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except Exception:
        raise ValueError(f"{name} could not be converted safely") from None
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must be a length-{length} vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def _response_matrix(values: object, n_persons: int, n_items: int) -> np.ndarray:
    """Return one row-major response matrix aligned with the design."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except Exception:
        raise ValueError("responses could not be converted safely") from None
    if array.ndim != 2 or array.shape != (n_persons, n_items):
        raise ValueError("responses must have shape (n_observations, n_items)")
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float64)


def _optional_offsets(values: object | None, n_persons: int) -> np.ndarray:
    """Return empty offsets or one finite person-level location vector."""
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return _finite_vector(values, "person_offsets", n_persons)


def _exact_positive_real(value: object, name: str) -> float:
    """Return one strictly positive finite real without Boolean coercion."""
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number greater than zero")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite real number greater than zero")
    return number


def _exact_device(value: object) -> str:
    """Return one supported compute-device label."""
    if type(value) is not str:
        raise ValueError("device must be one of 'cpu', 'gpu', or 'auto'")
    device = value.strip().casefold()
    if device not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be one of 'cpu', 'gpu', or 'auto'")
    return device


def estimate_crossed_person_effects(
    responses: object,
    design: ContextMembershipDesign,
    *,
    item_intercepts: object,
    item_slopes: object | None = None,
    person_offsets: object | None = None,
    prior_scale: object = 1.0,
    max_iter: object = 50,
    tol: object = 1e-8,
    worker_count: object = 1,
    device: object = "auto",
) -> CrossedPersonEffectResult:
    """Estimate crossed / multiple-membership person effects ``u_h``.

    The kernel is a Gaussian-prior MAP / Newton estimator of the Fox and Glas
    (2001) multilevel IRT group effects, using Browne, Goldstein, and Rasbash
    (2001) multiple-membership weights. Persons may belong to several units of
    one classification and to several classifications at once. Known item
    parameters stay fixed. Optional ``person_offsets`` accept already-estimated
    longitudinal locations; this function does not estimate OLS or AR states.

    Parameters
    ----------
    responses:
        Binary response matrix aligned with ``design.observation_ids`` on axis
        0 and items on axis 1. Non-finite or negative cells are missing.
    design:
        A package-built ``ContextMembershipDesign``. Tampered designs fail
        closed before native dispatch.
    item_intercepts:
        Known item intercepts ``b_i``, length ``n_items``.
    item_slopes:
        Known item discriminations ``a_i``. When omitted, Rasch slopes of 1
        are used.
    person_offsets:
        Optional person-level location offsets ``theta_p`` aligned with
        ``design.observation_ids``. Use this to consume a longitudinal state
        already estimated elsewhere. ``None`` treats every offset as zero.
    prior_scale:
        Level-2 standard deviation ``sigma_u`` of the Fox and Glas Gaussian
        prior. The kernel uses precision ``1 / sigma_u^2``.
    max_iter:
        Newton / IRLS iteration budget (exact built-in ``int``, ``>= 1``).
    tol:
        Absolute effect-step convergence tolerance.
    worker_count:
        Deterministic CPU worker count (exact built-in ``int``, ``>= 1``).
        The estimate does not depend on this value.
    device:
        ``cpu``, ``gpu``, or ``auto``. ``auto`` / ``gpu`` use the wgpu
        person-score kernel when an adapter is present and fall back to the
        f64 CPU reduction otherwise.

    Returns
    -------
    CrossedPersonEffectResult
        Centered ``u_h`` estimates, log-likelihood, and termination metadata.

    Raises
    ------
    ValueError
        If controls, shapes, or the design fail the package contract.
    KeyError
        Not used; membership identities come from the sealed design.

    Notes
    -----
    Recovered effects are centered to sum to zero inside each classification.
    The estimator is a MAP point method, not Fox and Glas Gibbs sampling and
    not a variance-component ML claim.
    """
    if type(design) is not ContextMembershipDesign:
        raise ValueError("design must be an exact ContextMembershipDesign")
    _ = design.design_fingerprint
    n_persons = len(design.observation_ids)
    n_effects = len(design.context_keys)
    trusted_max_iter = exact_integer(max_iter, "max_iter", minimum=1, maximum=10_000)
    trusted_workers = exact_integer(
        worker_count, "worker_count", minimum=1, maximum=10_000
    )
    trusted_tol = _exact_positive_real(tol, "tol")
    trusted_scale = _exact_positive_real(prior_scale, "prior_scale")
    trusted_device = _exact_device(device)
    try:
        intercept_count = int(np.asarray(item_intercepts, dtype=np.float64).shape[0])
    except Exception:
        raise ValueError("item_intercepts could not be converted safely") from None
    intercepts = _finite_vector(item_intercepts, "item_intercepts", intercept_count)
    n_items = int(intercepts.shape[0])
    if item_slopes is None:
        item_slopes = np.ones(n_items, dtype=np.float64)
    slopes = _finite_vector(item_slopes, "item_slopes", n_items)
    if np.any(slopes <= 0.0):
        raise ValueError("item_slopes must be strictly positive")
    y = _response_matrix(responses, n_persons, n_items)
    offsets = _optional_offsets(person_offsets, n_persons)
    (
        row_offsets,
        context_indices,
        weights,
        classification_offsets,
        context_keys,
    ) = _csr_from_design(design)
    core = multilevel_core()
    payload = core.estimate_crossed_person_effects(
        y,
        row_offsets,
        context_indices,
        weights,
        slopes,
        intercepts,
        offsets,
        classification_offsets,
        n_persons,
        n_items,
        n_effects,
        1.0 / (trusted_scale * trusted_scale),
        trusted_max_iter,
        trusted_tol,
        trusted_workers,
        trusted_device,
    )
    effect_vector = np.ascontiguousarray(payload["effects"], dtype=np.float64)
    context_effects = {
        key: float(value) for key, value in zip(context_keys, effect_vector, strict=True)
    }
    return CrossedPersonEffectResult(
        context_effects=context_effects,
        effect_vector=effect_vector,
        context_keys=context_keys,
        loglik=float(payload["loglik"]),
        n_iter=int(payload["n_iter"]),
        converged=bool(payload["converged"]),
        used_gpu=bool(payload["used_gpu"]),
        termination_reason=str(payload["termination_reason"]),
    )


__all__ = [
    "ContextKey",
    "CrossedPersonEffectResult",
    "estimate_crossed_person_effects",
    "weighted_contextual_effect",
]
