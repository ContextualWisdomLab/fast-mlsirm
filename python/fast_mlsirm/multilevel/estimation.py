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
from .contracts import ContextMembershipDesign, LongitudinalDesign, LongitudinalStateKind

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


def _observed_value(values: Mapping[str, float], occasion_id: str) -> float:
    """Return one caller observation as a real float, or NaN when absent."""
    try:
        raw = values.get(occasion_id, np.nan)
    except Exception:  # noqa: BLE001 - caller-controlled Mapping implementation
        raise ValueError("values must be a plain read-only mapping") from None
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
        raw, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"values[{occasion_id!r}] must be a real number")
    return float(raw)


def fit_longitudinal_state(
    design: LongitudinalDesign,
    values: Mapping[str, float],
    *,
    worker_count: int = 1,
) -> dict[str, object]:
    """Fit the Rust-owned respondent state predictor for a sealed design.

    ``values`` maps exact occasion identifiers to observed factor scores. A
    missing identifier is represented as ``NaN`` so the design remains intact
    while the Rust fitter excludes that observation from estimation. The
    returned state is aligned with ``design.occasions`` sorted by respondent
    and sequence, and includes normative estimand metadata so callers cannot
    mistake independent respondent OLS trends for population random effects or
    a caller-supplied AR coefficient for an estimated parameter.

    Raises
    ------
    ValueError
        If the execution controls, sealed design, caller observation mapping,
        or Rust-side state contract is invalid.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if type(design) is not LongitudinalDesign:
        raise ValueError("design must be an exact LongitudinalDesign")
    _ = design.design_fingerprint
    occasions = list(design.occasions)
    grouped: dict[str, list] = {
        respondent_id: [] for respondent_id in design.respondent_ids
    }
    for occasion in occasions:
        grouped[occasion.respondent_id].append(occasion)

    row_offsets = [0]
    sequence_indices: list[int] = []
    time_offsets: list[int] = []
    observations: list[float] = []
    for respondent_id in design.respondent_ids:
        for occasion in grouped[respondent_id]:
            sequence_indices.append(occasion.sequence_index)
            time_offsets.append(occasion.time_offset_milliseconds)
            observations.append(_observed_value(values, occasion.occasion_id))
        row_offsets.append(len(time_offsets))
    state_kind = design.state_spec.state_kind
    ar_coefficient = design.state_spec.autoregressive_coefficient
    if state_kind is LongitudinalStateKind.RANDOM_INTERCEPT_SLOPE:
        ar_coefficient = None
        estimand_scope = "independent_respondent_ols_trend"
        ar_coefficient_source = "not_applicable"
    else:
        estimand_scope = "discrete_ar_state_prediction"
        ar_coefficient_source = "caller_supplied"
    core = multilevel_core()
    result = core.fit_longitudinal_state(
        np.asarray(row_offsets, dtype=np.uint64),
        np.asarray(sequence_indices, dtype=np.uint64),
        np.asarray(time_offsets, dtype=np.int64),
        np.asarray(observations, dtype=np.float64),
        state_kind.value,
        ar_coefficient,
        worker_count,
    )
    return {
        "state_kind": state_kind.value,
        "estimand_scope": estimand_scope,
        "population_random_effects_estimated": False,
        "ar_coefficient_estimated": False,
        "ar_coefficient_source": ar_coefficient_source,
        "state_spec_fingerprint": design.state_spec.state_spec_fingerprint,
        "design_fingerprint": design.design_fingerprint,
        "state": np.asarray(result["state"], dtype=np.float64),
        "intercepts": np.asarray(result["intercepts"], dtype=np.float64),
        "slopes": np.asarray(result["slopes"], dtype=np.float64),
        "ar_coefficient": float(result["ar_coefficient"]),
        "rmse": float(result["rmse"]),
        "observed_count": int(result["observed_count"]),
        "transition_count": int(result["transition_count"]),
        "engine": str(result["engine"]),
        "respondent_ids": list(design.respondent_ids),
        "occasion_ids": [occasion.occasion_id for occasion in occasions],
        "occasion_records": [
            {
                "occasion_id": occasion.occasion_id,
                "respondent_id": occasion.respondent_id,
                "sequence_index": occasion.sequence_index,
                "time_offset_milliseconds": occasion.time_offset_milliseconds,
            }
            for occasion in occasions
        ],
    }


__all__ = ["ContextKey", "fit_longitudinal_state", "weighted_contextual_effect"]
