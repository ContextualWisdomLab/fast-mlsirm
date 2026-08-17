"""Typed Python access to the Rust-native contextual and longitudinal predictors.

This module performs marshalling only: converting a validated
``ContextMembershipDesign`` (see ``fast_mlsirm.multilevel.contracts``) into
the flat CSR arrays ``mlsirm_core::multilevel::weighted_contextual_effect``
expects, converting the caller's per-context random-effect values into the
matching flat vector, and converting a sealed ``LongitudinalDesign`` into the
row-offset arrays the Rust longitudinal kernels expect. The additive
contextual sum, independent per-respondent OLS trends, caller-supplied
discrete AR predictions, joint MAP hierarchical continuous-time AR(1) Rasch
estimation, worker determinism, and numerical input validation are owned by
the Rust core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .._multilevel_core_loader import multilevel_core
from .contracts import ContextMembershipDesign, LongitudinalDesign, LongitudinalStateKind

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


def _ordered_longitudinal_rows(design: LongitudinalDesign) -> tuple[list[int], list, list[int]]:
    """Return CSR offsets, ordered occasions, and millisecond offsets."""
    grouped: dict[str, list] = {
        respondent_id: [] for respondent_id in design.respondent_ids
    }
    for occasion in design.occasions:
        grouped[occasion.respondent_id].append(occasion)
    row_offsets = [0]
    time_offsets: list[int] = []
    ordered_occasions: list = []
    for respondent_id in design.respondent_ids:
        for occasion in grouped[respondent_id]:
            time_offsets.append(occasion.time_offset_milliseconds)
            ordered_occasions.append(occasion)
        row_offsets.append(len(time_offsets))
    return row_offsets, ordered_occasions, time_offsets


def _observed_value(values: Mapping[str, float], occasion_id: str) -> float:
    """Return one caller observation as a real float, or NaN when absent."""
    try:
        raw = values.get(occasion_id, np.nan)
    except Exception:
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

    Parameters
    ----------
    design:
        A package-built ``LongitudinalDesign``. Integrity is verified before
        marshalling so a tampered or hand-constructed design raises here.
    values:
        Mapping from occasion identifier to a real observed state. Absent
        keys become ``NaN``. Boolean, complex, and non-numeric values are
        rejected before native dispatch.
    worker_count:
        Number of deterministic worker threads (``>= 1``). The numerical
        result does not depend on this value.

    Returns
    -------
    dict
        Predicted states, respondent intercepts and slopes, RMSE, counts,
        engine identity, fingerprints, and normative estimand metadata.

    Raises
    ------
    ValueError
        If ``worker_count < 1``, the design is not an exact sealed
        ``LongitudinalDesign``, a caller observation cannot be read or
        converted safely, or the Rust-side state contract is invalid.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if type(design) is not LongitudinalDesign:
        raise ValueError("design must be an exact LongitudinalDesign")
    _ = design.design_fingerprint
    row_offsets, ordered_occasions, time_offsets = _ordered_longitudinal_rows(design)
    sequence_indices = [occasion.sequence_index for occasion in ordered_occasions]
    observations = [
        _observed_value(values, occasion.occasion_id) for occasion in ordered_occasions
    ]
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
        "occasion_ids": [occasion.occasion_id for occasion in ordered_occasions],
        "occasion_records": [
            {
                "occasion_id": occasion.occasion_id,
                "respondent_id": occasion.respondent_id,
                "sequence_index": occasion.sequence_index,
                "time_offset_milliseconds": occasion.time_offset_milliseconds,
            }
            for occasion in ordered_occasions
        ],
    }


def _validate_binary_response_matrix(
    responses: object,
    n_occasions: int,
) -> np.ndarray:
    """Return a C-contiguous float64 response matrix or a package-owned error."""
    if isinstance(responses, (bool, np.bool_)) or not isinstance(responses, np.ndarray):
        raise ValueError("responses must be a NumPy ndarray")
    if responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional occasion-by-item matrix")
    if responses.shape[0] != n_occasions:
        raise ValueError("responses rows must align with the sealed occasion order")
    if responses.shape[1] < 2:
        raise ValueError("hierarchical CT-AR Rasch requires at least two items")
    if responses.dtype == np.bool_ or np.issubdtype(responses.dtype, np.bool_):
        raise ValueError("responses must be 0, 1, or NaN rather than Boolean values")
    try:
        matrix = np.ascontiguousarray(responses, dtype=np.float64)
    except Exception:
        raise ValueError("responses could not be converted to float64 safely") from None
    finite = np.isfinite(matrix)
    invalid = finite & (matrix != 0.0) & (matrix != 1.0)
    if np.any(invalid):
        raise ValueError("responses must be 0, 1, or NaN")
    return matrix


def _item_labels(item_ids: Sequence[str] | None, n_items: int) -> list[str]:
    """Return caller item labels or stable positional defaults."""
    if item_ids is None:
        return [f"item_{index}" for index in range(n_items)]
    if isinstance(item_ids, (str, bytes)) or not isinstance(item_ids, Sequence):
        raise ValueError("item_ids must be a sequence of item identifiers")
    labels = list(item_ids)
    if len(labels) != n_items:
        raise ValueError("item_ids length must equal the response item axis")
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ValueError("item_ids entries must be non-empty strings")
    return labels


def fit_hierarchical_longitudinal_irt(
    design: LongitudinalDesign,
    responses: np.ndarray,
    *,
    item_ids: Sequence[str] | None = None,
    worker_count: int = 1,
    max_iter: int = 250,
    tolerance: float = 1e-5,
    hessian_step: float = 1e-3,
) -> dict[str, object]:
    """Fit the joint MAP hierarchical continuous-time AR(1) Rasch slice.

    This entry point does **not** interpret ``design.state_spec.state_kind``.
    The sealed design supplies respondent identity, occasion order, and exact
    millisecond offsets only. The estimand is joint MAP of a Rasch measurement
    model and a hierarchical stationary Ornstein–Uhlenbeck / continuous-time
    AR(1) latent-state process. It is not independent respondent OLS, not a
    caller-supplied discrete AR coefficient, not Fox and Glas (2001) Gibbs
    sampling, and not Jeon and Rabe-Hesketh (2016) adaptive-quadrature ML.

    Crossed and multiple-membership random effects are excluded from this
    joint likelihood. The existing GPU abstraction owns MLSIRM
    distance/likelihood kernels, not this hierarchical CT-AR Rasch objective,
    so ``gpu_parity`` is reported as false.

    Parameters
    ----------
    design:
        A package-built ``LongitudinalDesign``. Integrity is verified before
        marshalling so a tampered or hand-constructed design raises here.
    responses:
        Occasion-major binary matrix with shape ``(n_occasions, n_items)``,
        aligned with ``design.occasions`` after respondent-then-sequence
        ordering. Values must be ``0``, ``1``, or ``NaN``.
    item_ids:
        Optional item labels aligned with the response columns. Defaults to
        ``item_0``, ``item_1``, ...
    worker_count:
        Number of deterministic person-shard worker threads (``>= 1``).
    max_iter:
        Maximum packed L-BFGS iterations (``>= 1``).
    tolerance:
        Relative L-BFGS tolerance; must be finite and strictly positive.
    hessian_step:
        Central-difference step for the hyperparameter observed Hessian.

    Returns
    -------
    dict
        Joint MAP states, Wald intervals, item intercepts, estimated
        population mean/sd/decay, unit-day AR coefficient, counts, engine
        identity, fingerprints, and normative estimand metadata.

    Raises
    ------
    ValueError
        If execution controls, the sealed design, or the response matrix are
        invalid, or the Rust kernel rejects the design.
    """
    if worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")
    if not np.isfinite(hessian_step) or hessian_step <= 0.0:
        raise ValueError("hessian_step must be finite and strictly positive")
    if type(design) is not LongitudinalDesign:
        raise ValueError("design must be an exact LongitudinalDesign")
    _ = design.design_fingerprint
    row_offsets, ordered_occasions, time_offsets = _ordered_longitudinal_rows(design)
    matrix = _validate_binary_response_matrix(responses, len(ordered_occasions))
    labels = _item_labels(item_ids, matrix.shape[1])
    core = multilevel_core()
    result = core.fit_hierarchical_ctar_rasch(
        np.asarray(row_offsets, dtype=np.uint64),
        np.asarray(time_offsets, dtype=np.int64),
        matrix,
        worker_count,
        max_iter,
        float(tolerance),
        float(hessian_step),
    )
    return {
        "estimand_scope": str(result["estimand_scope"]),
        "transition_kind": str(result["transition_kind"]),
        "interval_kind": str(result["interval_kind"]),
        "population_random_effects_estimated": True,
        "ar_coefficient_estimated": True,
        "ar_coefficient_source": "joint_map",
        "multiple_membership_estimated": False,
        "gpu_parity": False,
        "state_spec_fingerprint": design.state_spec.state_spec_fingerprint,
        "design_fingerprint": design.design_fingerprint,
        "state": np.asarray(result["state"], dtype=np.float64),
        "state_se": np.asarray(result["state_se"], dtype=np.float64),
        "state_lower": np.asarray(result["state_lower"], dtype=np.float64),
        "state_upper": np.asarray(result["state_upper"], dtype=np.float64),
        "item_intercepts": np.asarray(result["item_intercepts"], dtype=np.float64),
        "item_ids": labels,
        "population_mean": float(result["population_mean"]),
        "population_sd": float(result["population_sd"]),
        "decay_rate": float(result["decay_rate"]),
        "unit_time_ar_coefficient": float(result["unit_time_ar_coefficient"]),
        "hyperparameter_se": np.asarray(result["hyperparameter_se"], dtype=np.float64),
        "hyperparameter_lower": np.asarray(result["hyperparameter_lower"], dtype=np.float64),
        "hyperparameter_upper": np.asarray(result["hyperparameter_upper"], dtype=np.float64),
        "hyperparameter_intervals_identified": bool(
            result["hyperparameter_intervals_identified"]
        ),
        "state_intervals_identified": bool(result["state_intervals_identified"]),
        "observed_count": int(result["observed_count"]),
        "transition_count": int(result["transition_count"]),
        "status": str(result["status"]),
        "engine": str(result["engine"]),
        "respondent_ids": list(design.respondent_ids),
        "occasion_ids": [occasion.occasion_id for occasion in ordered_occasions],
        "occasion_records": [
            {
                "occasion_id": occasion.occasion_id,
                "respondent_id": occasion.respondent_id,
                "sequence_index": occasion.sequence_index,
                "time_offset_milliseconds": occasion.time_offset_milliseconds,
            }
            for occasion in ordered_occasions
        ],
    }


def simulate_hierarchical_longitudinal_irt(
    design: LongitudinalDesign,
    *,
    item_intercepts: Sequence[float],
    population_mean: float = 0.0,
    population_sd: float = 0.7,
    decay_rate: float = 0.35,
    seed: int = 1,
) -> dict[str, object]:
    """Simulate hierarchical CT-AR Rasch states and binary responses.

    The simulator is the recovery-fixture generator for
    ``fit_hierarchical_longitudinal_irt``. It is not a claim that the
    subsequent fit recovers these parameters without shrinkage.

    Parameters
    ----------
    design:
        A package-built ``LongitudinalDesign`` supplying occasion times.
    item_intercepts:
        Generating Rasch item intercepts.
    population_mean:
        Generating population mean of the latent-state process.
    population_sd:
        Generating stationary standard deviation.
    decay_rate:
        Generating continuous-time decay rate per day.
    seed:
        Deterministic unsigned seed forwarded to the Rust LCG.

    Returns
    -------
    dict
        Generating states and an occasion-major response matrix aligned with
        the sealed design order.

    Raises
    ------
    ValueError
        If the design or generating parameters are invalid.
    """
    if type(design) is not LongitudinalDesign:
        raise ValueError("design must be an exact LongitudinalDesign")
    _ = design.design_fingerprint
    if isinstance(item_intercepts, (str, bytes)) or not isinstance(
        item_intercepts, Sequence
    ):
        raise ValueError("item_intercepts must be a sequence of real numbers")
    try:
        intercepts = np.asarray(list(item_intercepts), dtype=np.float64)
    except Exception:
        raise ValueError("item_intercepts could not be converted safely") from None
    if intercepts.ndim != 1 or intercepts.size < 2:
        raise ValueError("item_intercepts must contain at least two finite values")
    if not np.all(np.isfinite(intercepts)):
        raise ValueError("item_intercepts must be finite")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    row_offsets, ordered_occasions, time_offsets = _ordered_longitudinal_rows(design)
    core = multilevel_core()
    result = core.simulate_hierarchical_ctar_rasch(
        np.asarray(row_offsets, dtype=np.uint64),
        np.asarray(time_offsets, dtype=np.int64),
        intercepts,
        float(population_mean),
        float(population_sd),
        float(decay_rate),
        int(seed),
    )
    n_items = int(result["n_items"])
    responses = np.asarray(result["responses"], dtype=np.float64).reshape(
        (len(ordered_occasions), n_items)
    )
    return {
        "state": np.asarray(result["state"], dtype=np.float64),
        "responses": responses,
        "item_intercepts": intercepts,
        "population_mean": float(population_mean),
        "population_sd": float(population_sd),
        "decay_rate": float(decay_rate),
        "occasion_ids": [occasion.occasion_id for occasion in ordered_occasions],
        "design_fingerprint": design.design_fingerprint,
    }


__all__ = [
    "ContextKey",
    "fit_hierarchical_longitudinal_irt",
    "fit_longitudinal_state",
    "simulate_hierarchical_longitudinal_irt",
    "weighted_contextual_effect",
]
