"""Provider-neutral decision-support arithmetic backed by Rust.

The public function accepts caller-supplied state probabilities, utilities, and
intervention costs.  It computes expected net intervention value, expected
value of perfect information (EVPI), and, when supplied, expected value of
sample information (EVSI).  It does not infer a utility function, learn a
probability from text, or claim a causal intervention effect.

The finite-table equations follow Howard (1966) and Raiffa and Schlaifer
(1961).  Python validates and marshals the table; all expected-value
arithmetic is performed in the Rust core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ata import _preflight_real_evidence, _trusted_real_array
from .utility import _coerce_finite_real

__all__ = ["DecisionSupportResult", "evaluate_decision_support"]

MAX_DECISION_ACTIONS = 1024
MAX_DECISION_STATES = 4096
MAX_DECISION_SIGNALS = 1024
MAX_DECISION_CELLS = 1_000_000

_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)


@dataclass
class DecisionSupportResult:
    """Rust-computed expected-value results for one explicit decision table.

    ``action_expected_net_values`` is in the caller's action-row order.
    ``selected_action`` is the first action attaining the greatest prior
    expected net value.  The information values are ``None`` when no joint
    state/signal distribution was supplied.
    """

    action_expected_net_values: np.ndarray
    selected_action: int
    expected_net_intervention_value: float
    expected_value_perfect_information: float
    expected_value_sample_information: float | None
    net_expected_value_sample_information: float | None


def _trusted_index(value: object, *, name: str) -> int:
    """Return a non-negative action index without caller conversion callbacks."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be a non-negative integer")
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _real_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    """Return bounded, lossless, contiguous binary64 evidence for Rust."""
    shape, cells = _preflight_real_evidence(value, name)
    if len(shape) != ndim:
        raise ValueError(f"{name} must be a {ndim}D real numeric array")
    if cells == 0:
        raise ValueError(f"{name} must not be empty")
    if cells > MAX_DECISION_CELLS:
        raise ValueError(f"{name} exceeds {MAX_DECISION_CELLS} cells")
    if type(value) is np.ndarray and value.dtype.kind == "b":
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    array = _trusted_real_array(value, name)
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    return np.ascontiguousarray(array, dtype=np.float64)


def _core_module():
    """Return the compiled Rust extension for the decision-support call."""
    from . import _core

    return _core


def evaluate_decision_support(
    state_probabilities: object,
    action_utilities: object,
    intervention_costs: object,
    no_action_index: object = 0,
    signal_joint_probabilities: object | None = None,
    information_cost: object = 0.0,
) -> DecisionSupportResult:
    """Evaluate explicit expected net intervention and information values.

    Parameters
    ----------
    state_probabilities:
        One-dimensional prior probabilities over finite states.  They must be
        non-negative and sum to one within the Rust binary64 tolerance.
    action_utilities:
        Two-dimensional ``(actions, states)`` utility table.  Utilities are
        caller-supplied consequences; they are not learned by this function.
    intervention_costs:
        One non-negative cost per action.  The designated no-action row must
        have zero cost.
    no_action_index:
        Row identifying the no-intervention baseline.
    signal_joint_probabilities:
        Optional ``(signals, states)`` joint ``P(signal, state)`` table.  Its
        state marginal must equal ``state_probabilities``; passing arbitrary
        posterior rows is rejected.
    information_cost:
        Non-negative cost subtracted from EVSI to form its net value.

    Notes
    -----
    For action ``a`` and state ``s``, the Rust core evaluates
    ``U(a, s) - U(no_action, s) - cost(a)``.  This is decision analysis over
    supplied inputs, not a causal estimate or an automated high-stakes policy.
    """
    probabilities = _real_array(
        state_probabilities,
        name="state_probabilities",
        ndim=1,
    )
    utilities = _real_array(action_utilities, name="action_utilities", ndim=2)
    costs = _real_array(intervention_costs, name="intervention_costs", ndim=1)
    action_index = _trusted_index(no_action_index, name="no_action_index")
    info_cost = _coerce_finite_real(information_cost, name="information_cost")

    state_count = probabilities.shape[0]
    action_count = utilities.shape[0]
    if state_count > MAX_DECISION_STATES:
        raise ValueError(f"state_probabilities exceeds {MAX_DECISION_STATES} states")
    if action_count > MAX_DECISION_ACTIONS:
        raise ValueError(f"action_utilities exceeds {MAX_DECISION_ACTIONS} actions")
    if utilities.shape[1] != state_count:
        raise ValueError("action_utilities columns must match state_probabilities")
    if costs.shape[0] != action_count:
        raise ValueError("intervention_costs length must match action_utilities rows")
    if action_index >= action_count:
        raise ValueError("no_action_index must identify one action")

    if signal_joint_probabilities is None:
        signals = None
        signal_count = 0
    else:
        signals = _real_array(
            signal_joint_probabilities,
            name="signal_joint_probabilities",
            ndim=2,
        )
        signal_count = signals.shape[0]
        if signal_count > MAX_DECISION_SIGNALS:
            raise ValueError(
                f"signal_joint_probabilities exceeds {MAX_DECISION_SIGNALS} signals"
            )
        if signals.shape[1] != state_count:
            raise ValueError(
                "signal_joint_probabilities columns must match state_probabilities"
            )

    result = _core_module().evaluate_decision_support(
        probabilities,
        utilities,
        costs,
        action_index,
        signals,
        info_cost,
    )
    return DecisionSupportResult(
        action_expected_net_values=np.asarray(
            result["action_expected_net_values"], dtype=np.float64
        ),
        selected_action=int(result["selected_action"]),
        expected_net_intervention_value=float(
            result["expected_net_intervention_value"]
        ),
        expected_value_perfect_information=float(
            result["expected_value_perfect_information"]
        ),
        expected_value_sample_information=(
            None
            if result["expected_value_sample_information"] is None
            else float(result["expected_value_sample_information"])
        ),
        net_expected_value_sample_information=(
            None
            if result["net_expected_value_sample_information"] is None
            else float(result["net_expected_value_sample_information"])
        ),
    )
