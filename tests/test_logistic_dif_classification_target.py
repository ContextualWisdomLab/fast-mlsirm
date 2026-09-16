"""Pin which quantity ``jg_class`` is computed from, and what the other one gives.

`logistic_dif` classifies items by comparing ``delta_r2`` -- a Nagelkerke
pseudo-R-squared change across the two-degree-of-freedom omnibus -- to the
Jodoin-Gierl ``.035`` / ``.070`` boundaries, and reports ``delta_r2_uniform``
without a class.

Jodoin & Gierl (2001), *Applied Measurement in Education, 14*(4), 329-349, has
since been read and states the bands on the ONE-degree-of-freedom UNIFORM
increment (p. 335). So the applied quantity is the one the bands were not built
for, and the unclassified one is the one they were. Correcting that reclassifies
every item every current caller has scored.

These tests change nothing. They record what the package does today next to what
the other quantity would give, so the correction lands as a visible diff in a
test file rather than as a silent change in numbers nobody re-derives. They are
expected to FAIL, loudly and by design, when the applied quantity changes -- at
which point the failing assertion names what to update and why.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import logistic_dif

N_PER_GROUP = 1_500
N_ITEMS = 8
CROSSING_ITEM = 3
INTERCEPT_SHIFT = 1.5
SLOPE_DIFFERENCE = 2.0
SEED = 20_260_915
# The Jodoin-Gierl boundaries as the package applies them.
MODERATE = 0.035
LARGE = 0.070


def _band(value: float) -> str:
    """The letter the boundaries give a value, ignoring the significance gate."""
    if not np.isfinite(value):
        return "U"
    if value < MODERATE:
        return "A"
    if value < LARGE:
        return "B"
    return "C"


@pytest.fixture(scope="module")
def sweep():
    """One CROSSING item: a group intercept shift AND a group slope difference.

    The fixture is crossing rather than purely uniform because that is the only
    place the two quantities separate. Under pure uniform differential
    functioning the interaction term contributes almost nothing and the omnibus
    and uniform increments nearly coincide -- measured here at 0.0523 against
    0.0503, the same band. The reclassification therefore bites on crossing
    items specifically, which are also the items the bands were least licensed
    for: the original studied only two of them and called the extension
    provisional.
    """
    rng = np.random.default_rng(SEED)
    theta = rng.standard_normal(2 * N_PER_GROUP)
    group = np.repeat([0, 1], N_PER_GROUP)
    focal = group == 1
    columns = []
    for item in range(N_ITEMS):
        slope = 1.2 + (SLOPE_DIFFERENCE * focal if item == CROSSING_ITEM else 0.0)
        intercept = INTERCEPT_SHIFT * focal if item == CROSSING_ITEM else 0.0
        p = 1.0 / (1.0 + np.exp(-(slope * theta + intercept)))
        columns.append((rng.random(theta.size) < p).astype(np.int64))
    return logistic_dif(np.column_stack(columns), group)


def test_the_class_is_computed_from_the_omnibus_quantity(sweep) -> None:
    """What ships today."""
    for item in range(N_ITEMS):
        if sweep["jg_class"][item] == "U":
            continue
        if not sweep["flagged_bh"][item]:
            # A non-significant item is forced to "A" regardless of magnitude,
            # so it cannot distinguish the two quantities.
            continue
        assert sweep["jg_class"][item] == _band(float(sweep["delta_r2"][item])), (
            f"item {item}: jg_class is not the band of delta_r2 -- if this now "
            "reads from delta_r2_uniform, the applied quantity has been "
            "corrected and this test should be replaced, not relaxed"
        )


def test_the_two_quantities_are_not_interchangeable(sweep) -> None:
    """The uniform increment is a different number, not a rounding of the same one."""
    finite = np.isfinite(sweep["delta_r2"]) & np.isfinite(sweep["delta_r2_uniform"])
    assert finite.any()
    assert not np.allclose(
        sweep["delta_r2"][finite], sweep["delta_r2_uniform"][finite], atol=1e-6
    )


def test_at_least_one_item_would_be_classified_differently(sweep) -> None:
    """The reclassification is real, not hypothetical.

    If this ever passes vacuously -- no item differing -- the fixture has lost
    the property it was built for and must be rebuilt, because a correction with
    no observable consequence cannot be reviewed.
    """
    differing = [
        item
        for item in range(N_ITEMS)
        if np.isfinite(sweep["delta_r2"][item])
        and np.isfinite(sweep["delta_r2_uniform"][item])
        and _band(float(sweep["delta_r2"][item]))
        != _band(float(sweep["delta_r2_uniform"][item]))
    ]

    assert CROSSING_ITEM in differing, (
        "the crossing item's band no longer differs between the omnibus and "
        "uniform quantities, so this fixture can no longer demonstrate the "
        "reclassification and must be rebuilt"
    )
    # Today: omnibus 0.043 -> "B", uniform 0.027 -> "A". Correcting the applied
    # quantity moves this item down a band.
    assert _band(float(sweep["delta_r2"][CROSSING_ITEM])) == "B"
    assert _band(float(sweep["delta_r2_uniform"][CROSSING_ITEM])) == "A"


def test_the_uniform_increment_is_reported_without_a_class(sweep) -> None:
    """Today's contract: the quantity the bands were built for carries no class."""
    assert "delta_r2_uniform" in sweep
    assert "jg_class_uniform" not in sweep
