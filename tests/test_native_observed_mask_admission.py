"""Direct compiled binding admission; deliberate map refusal prevents fitting.

These tests exercise buffer safety, not numerical or hardware acceptance.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import _core

_ENTRIES = (
    "fit_bifactor_grm",
    "fit_bifactor_grm_multigroup",
    "fit_bifactor_grm_fipc",
    "fit_two_tier_grm",
    "bifactor_oakes_se",
    "two_tier_oakes_se",
)


def _call(entry, y, observed, n_persons=1, n_items=1):
    """Invoke the native converter with an intentional pre-fit i32 barrier."""
    common = dict(
        y=np.asarray(y, dtype=np.int64),
        observed=None if observed is None else np.asarray(observed, dtype=np.bool_),
        specific_map=np.array([2**40], dtype=np.int64),
        n_persons=n_persons,
        n_items=n_items,
        n_specific=1,
        n_cat=2,
    )
    one = np.ones(1, dtype=np.float64)
    zero = np.zeros(1, dtype=np.float64)
    boolean = np.ones(1, dtype=np.bool_)
    if entry == "fit_bifactor_grm":
        extra = dict(q_general=121, q_specific=121, device="cpu")
    elif entry == "fit_bifactor_grm_multigroup":
        extra = dict(group_id=np.zeros(1, dtype=np.int64), n_groups=1,
                     q_general=121, q_specific=121, device="cpu")
    elif entry == "fit_bifactor_grm_fipc":
        extra = dict(anchor=boolean, fixed_a_general=one, fixed_a_specific=one,
                     fixed_threshold=zero, q_general=121, q_specific=121)
    elif entry == "fit_two_tier_grm":
        extra = dict(primary_map=boolean, n_primary=1, q_primary=121,
                     q_specific=121, primary_correlation="identity")
    elif entry == "bifactor_oakes_se":
        extra = dict(a_general=one, a_specific=one, threshold=zero,
                     q_general=121, q_specific=121, fd_step=1e-4)
    else:
        extra = dict(a_primary=one, a_specific=one, threshold=zero, phi=one,
                     primary_map=boolean, n_primary=1, q_primary=121,
                     q_specific=121, fd_step=1e-4, primary_correlation="identity")
    return getattr(_core, entry)(**common, **extra)


@pytest.mark.parametrize("entry", _ENTRIES)
def test_short_mask_is_valueerror_before_negative_placeholder_index(entry):
    """Reject an empty supplied mask instead of raising a PyO3 panic."""
    with pytest.raises(ValueError, match="observed must have length n_persons \\* n_items"):
        _call(entry, [-1], [])


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("y,mask", [([0], []), ([-1], [False, False])])
def test_mask_length_is_checked_for_all_response_signs(entry, y, mask):
    """Reject short or long masks independently of response values."""
    with pytest.raises(ValueError, match="observed must have length n_persons \\* n_items"):
        _call(entry, y, mask)


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("persons,items", [(0, 1), (2, 1), (2**63, 2)])
def test_response_dimensions_are_checked_before_conversion(entry, persons, items):
    """Reject zero, inconsistent and overflowing shapes without allocation."""
    with pytest.raises(ValueError, match="n_persons|y must have length"):
        _call(entry, [-1], [], persons, items)


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("y,mask", [([-1], [False]), ([0], [True]), ([0], None)])
def test_valid_conversion_reaches_map_barrier_without_fit(entry, y, mask):
    """Preserve valid masked placeholders and nonnegative conversion."""
    with pytest.raises(ValueError, match="specific_map entries must fit in i32"):
        _call(entry, y, mask)


@pytest.mark.parametrize("entry", _ENTRIES)
def test_observed_negative_remains_rejected(entry):
    """Never silently normalize an observed negative response."""
    with pytest.raises(ValueError, match="y categories must be non-negative"):
        _call(entry, [-1], [True])


@pytest.mark.parametrize("entry", _ENTRIES)
def test_absent_mask_negative_policy_is_preserved(entry):
    """Preserve the published observed-negative policy without dropping geometry cases."""
    message = ("non-negative when observed is None" if "oakes" in entry
               else "y categories must be non-negative")
    with pytest.raises(ValueError, match=message):
        _call(entry, [-1], None)
