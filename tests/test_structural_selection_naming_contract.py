"""Naming-contract regressions for structural-selection helper boundaries."""

from inspect import signature

from fast_mlsirm.structural_selection import _require_candidate_id, _require_exact_bool


def test_structural_helper_parameters_are_semantically_specific() -> None:
    """Private validation helpers must describe the bounded-context values they validate."""
    candidate_parameter_names = tuple(signature(_require_candidate_id).parameters)
    policy_parameter_names = tuple(signature(_require_exact_bool).parameters)

    assert candidate_parameter_names == ("candidate_value", "candidate_role_name")
    assert policy_parameter_names == ("policy_value", "policy_fact_name")
    assert not {"value", "name"} & set(candidate_parameter_names)
    assert not {"value", "name"} & set(policy_parameter_names)
