"""Typed Python access to the Rust-native contextual-effects predictor.

This module performs marshalling only: converting a validated
``ContextMembershipDesign`` (see ``fast_mlsirm.multilevel.contracts``) into
the flat CSR arrays ``mlsirm_core::multilevel::weighted_contextual_effect``
expects, and converting the caller's per-context random-effect values into
the matching flat vector. The additive sum, its determinism across worker
counts, and its input validation are owned by the Rust core; see that
module's docstring for the full linear-predictor context and the
Browne, Goldstein, and Rasbash (2001) citation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .._multilevel_core_loader import multilevel_core
from .contracts import ContextMembershipDesign

ContextKey = tuple[str, str]


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
        key ``design`` references.
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
        If ``worker_count < 1``, or the design fails integrity verification.
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
    missing = [key for key in context_keys if key not in context_effects]
    if missing:
        raise KeyError(f"context_effects is missing keys: {missing!r}")
    key_index = {key: index for index, key in enumerate(context_keys)}
    effects = np.array(
        [float(context_effects[key]) for key in context_keys], dtype=np.float64
    )

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


__all__ = ["ContextKey", "weighted_contextual_effect"]
