"""Real PyO3 response admission with a zero-budget sentinel; no fit executes."""
import numpy as np
import pytest
from fast_mlsirm import _core


def _call(y, observed, family="fit_two_tier_grm_fipc"):
    """Stop valid conversions at config validation before quadrature or E-step."""
    args = dict(
        y=np.array(y, dtype=np.int64), observed=observed,
        specific_map=np.array([0, 0], dtype=np.int64),
        n_persons=2, n_items=2, n_specific=1, n_cat=2,
        max_iter=0, tol=1e-6,
    )
    if "two_tier" in family:
        args.update(primary_map=np.array([True, True]), n_primary=1,
                    q_primary=121, q_specific=121)
    else:
        args.update(q_general=121, q_specific=121)
    if family.endswith("multigroup"):
        args.update(group_id=np.array([0, 0], dtype=np.int64), n_groups=1)
    if family.endswith("fipc"):
        args.update(anchor=np.ones(2, dtype=bool),
                    fixed_a_specific=np.array([0.5, 0.5]),
                    fixed_threshold=np.array([0.1, -0.1]))
        key = "fixed_a_primary" if "two_tier" in family else "fixed_a_general"
        args[key] = np.array([1.0, 1.0])
    return getattr(_core, family)(**args)


FAMILIES = ["fit_bifactor_grm", "fit_bifactor_grm_multigroup",
            "fit_bifactor_grm_fipc", "fit_two_tier_grm", "fit_two_tier_grm_fipc"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("observed", [None, np.ones(4, dtype=bool)], ids=["no-mask", "all-observed"])
def test_negative_observed_response_is_rejected_before_zero_budget(observed, family):
    """Without a mask, all responses are observed and cannot be negative."""
    with pytest.raises(ValueError, match="y categories must be non-negative"):
        _call([-1, 1, 1, 0], observed, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_explicitly_missing_negative_reaches_zero_budget_sentinel(family):
    """A false mask entry preserves supported missing-response conversion."""
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        _call([-1, 1, 1, 0], np.array([False, True, True, True]), family)


@pytest.mark.parametrize("family", FAMILIES)
def test_complete_nonnegative_data_reaches_zero_budget_sentinel(family):
    """A valid complete response matrix retains its input-admission path."""
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        _call([0, 1, 1, 0], None, family)
