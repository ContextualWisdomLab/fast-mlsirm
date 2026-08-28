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
from math import isfinite

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
_RESULT_KEYS = frozenset(
    {
        "action_expected_net_values",
        "selected_action",
        "expected_net_intervention_value",
        "expected_value_perfect_information",
        "expected_value_sample_information",
        "net_expected_value_sample_information",
    }
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


def _validated_rust_result(
    result: object,
    *,
    action_count: int,
    has_sample_information: bool,
    information_cost: float = 0.0,
) -> tuple[list[float], int, float, float, float | None, float | None]:
    """Replay the concrete PyO3 result contract before public marshalling."""
    invalid = RuntimeError("invalid decision-support Rust result payload")
    if type(result) is not dict:
        raise invalid
    keys = list(dict.keys(result))
    if any(type(key) is not str for key in keys) or set(keys) != _RESULT_KEYS:
        raise invalid

    action_values = dict.__getitem__(result, "action_expected_net_values")
    selected_action = dict.__getitem__(result, "selected_action")
    expected_net = dict.__getitem__(result, "expected_net_intervention_value")
    evpi = dict.__getitem__(result, "expected_value_perfect_information")
    evsi = dict.__getitem__(result, "expected_value_sample_information")
    net_evsi = dict.__getitem__(result, "net_expected_value_sample_information")

    if type(action_values) is not list or len(action_values) != action_count:
        raise invalid
    if any(type(value) is not float or not isfinite(value) for value in action_values):
        raise invalid
    if type(selected_action) is not int or not 0 <= selected_action < action_count:
        raise invalid
    if type(expected_net) is not float or not isfinite(expected_net):
        raise invalid
    if type(evpi) is not float or not isfinite(evpi):
        raise invalid
    if expected_net != action_values[selected_action]:
        raise invalid
    first_best = 0
    for index in range(1, action_count):
        if action_values[index] > action_values[first_best]:
            first_best = index
    if selected_action != first_best:
        raise invalid

    if has_sample_information:
        if type(evsi) is not float or not isfinite(evsi):
            raise invalid
        if type(net_evsi) is not float or not isfinite(net_evsi):
            raise invalid
        if net_evsi != evsi - information_cost:
            raise invalid
    elif evsi is not None or net_evsi is not None:
        raise invalid

    return action_values, selected_action, expected_net, evpi, evsi, net_evsi


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

    native_result = _core_module().evaluate_decision_support(
        probabilities,
        utilities,
        costs,
        action_index,
        signals,
        info_cost,
    )
    (
        action_values,
        selected_action,
        expected_net,
        evpi,
        evsi,
        net_evsi,
    ) = _validated_rust_result(
        native_result,
        action_count=action_count,
        has_sample_information=signals is not None,
        information_cost=info_cost,
    )
    return DecisionSupportResult(
        action_expected_net_values=np.array(action_values, dtype=np.float64),
        selected_action=selected_action,
        expected_net_intervention_value=expected_net,
        expected_value_perfect_information=evpi,
        expected_value_sample_information=evsi,
        net_expected_value_sample_information=net_evsi,
    )
