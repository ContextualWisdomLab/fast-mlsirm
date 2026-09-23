"""Contract for consuming a fitted multiple-group bifactor GRM as expected raw
scores.

:class:`~fast_mlsirm.bifactor_multigroup.BifactorMultigroupFit` stores its item
parameters with a leading group axis (``n_groups x n_items``), which the
single-group expected-score path could not read. The group axis carries
separate item parameters whenever any item is free to differ across groups, so
it is *selected* here, never flattened: a group must be named unless every
group's rows are exactly equal, the identity an all-anchored fit
(``anchor_mask=None``) creates by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import (
    check_bifactor_expected_total_score_monotonicity,
    predict_bifactor_expected_total_score,
)

N_CAT = 4
THRESHOLDS = np.array([1.2, 0.0, -1.2])
A_GENERAL = np.array([1.1, 0.9, 1.3, 0.7])
A_SPECIFIC = np.array([0.6, 0.5, -0.4, 0.8])


class _SingleGroupFit:
    """The fields the expected-score path reads from a single-group fit."""

    def __init__(self, a_general: np.ndarray, a_specific: np.ndarray) -> None:
        self.a_general = np.asarray(a_general, dtype=np.float64)
        self.a_specific = np.asarray(a_specific, dtype=np.float64)
        self.threshold = np.tile(THRESHOLDS, (self.a_general.shape[0], 1))
        self.n_cat = N_CAT


class _MultigroupFit:
    """The fields the expected-score path reads from a multiple-group fit."""

    def __init__(
        self,
        a_general: np.ndarray,
        a_specific: np.ndarray,
        specific_sd: np.ndarray | None = None,
    ) -> None:
        self.a_general = np.asarray(a_general, dtype=np.float64)
        self.a_specific = np.asarray(a_specific, dtype=np.float64)
        n_groups, n_items = self.a_general.shape
        self.threshold = np.tile(THRESHOLDS, (n_groups, n_items, 1))
        self.specific_sd = (
            np.ones((n_groups, 1)) if specific_sd is None else np.asarray(specific_sd)
        )
        self.n_cat = N_CAT
        self.n_groups = n_groups


def _anchored(n_groups: int = 3) -> _MultigroupFit:
    """An all-anchored fit: every group's rows identical by construction."""
    return _MultigroupFit(
        np.tile(A_GENERAL, (n_groups, 1)), np.tile(A_SPECIFIC, (n_groups, 1))
    )


def test_anchored_fit_matches_the_single_group_curve() -> None:
    """Identical rows mean the multiple-group curve is the single-group one."""
    grid = np.linspace(-2.5, 2.5, 11)
    single = predict_bifactor_expected_total_score(
        _SingleGroupFit(A_GENERAL, A_SPECIFIC), grid, q_specific=41
    )

    fit = _anchored()
    for group in (None, 0, 1, 2):
        got = predict_bifactor_expected_total_score(
            fit, grid, q_specific=41, group=group
        )
        np.testing.assert_allclose(got, single, atol=1e-12)


def test_group_specific_rows_require_naming_the_group() -> None:
    """No group is picked silently when the rows are not the same items."""
    a_general = np.vstack([A_GENERAL, A_GENERAL * 0.5])
    fit = _MultigroupFit(a_general, np.tile(A_SPECIFIC, (2, 1)))
    grid = np.linspace(-2.0, 2.0, 9)

    with pytest.raises(ValueError, match="group="):
        predict_bifactor_expected_total_score(fit, grid, q_specific=41)

    first = predict_bifactor_expected_total_score(fit, grid, q_specific=41, group=0)
    second = predict_bifactor_expected_total_score(fit, grid, q_specific=41, group=1)
    assert not np.allclose(first, second)
    np.testing.assert_allclose(
        first,
        predict_bifactor_expected_total_score(
            _SingleGroupFit(A_GENERAL, A_SPECIFIC), grid, q_specific=41
        ),
        atol=1e-12,
    )


def test_threshold_rows_are_part_of_the_identity_check() -> None:
    """Identity covers every item parameter, not the slopes alone."""
    fit = _anchored(2)
    fit.threshold = fit.threshold.copy()
    fit.threshold[1, 0] = THRESHOLDS + 0.3
    with pytest.raises(ValueError, match="group="):
        predict_bifactor_expected_total_score(
            fit, np.linspace(-2.0, 2.0, 5), q_specific=41
        )


def test_rejects_estimated_specific_sd() -> None:
    """An estimated specific-factor SD needs a specific_map the fit lacks."""
    fit = _MultigroupFit(
        np.tile(A_GENERAL, (2, 1)),
        np.tile(A_SPECIFIC, (2, 1)),
        specific_sd=np.array([[1.0], [1.4]]),
    )
    grid = np.linspace(-2.0, 2.0, 5)
    with pytest.raises(ValueError, match="specific_sd"):
        predict_bifactor_expected_total_score(fit, grid, q_specific=41)
    with pytest.raises(ValueError, match="specific_sd"):
        predict_bifactor_expected_total_score(fit, grid, q_specific=41, group=1)
    # Group 0's own SD is 1, so that group stays computable.
    predict_bifactor_expected_total_score(fit, grid, q_specific=41, group=0)


@pytest.mark.parametrize("group", [-1, 3, 1.5])
def test_rejects_out_of_range_group(group: object) -> None:
    with pytest.raises(ValueError):
        predict_bifactor_expected_total_score(
            _anchored(), np.linspace(-2.0, 2.0, 5), q_specific=41, group=group
        )


def test_group_is_rejected_for_a_single_group_fit() -> None:
    with pytest.raises(ValueError, match="multiple-group"):
        predict_bifactor_expected_total_score(
            _SingleGroupFit(A_GENERAL, A_SPECIFIC),
            np.linspace(-2.0, 2.0, 5),
            q_specific=41,
            group=0,
        )


def test_pointwise_theta_may_tie_and_be_unordered() -> None:
    """Person EAPs are not a grid: ties and any order evaluate pointwise."""
    grid = np.array([-1.0, 0.0, 1.0])
    curve = predict_bifactor_expected_total_score(_anchored(), grid, q_specific=41)

    theta = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
    got = predict_bifactor_expected_total_score(_anchored(), theta, q_specific=41)
    np.testing.assert_allclose(got, curve[[2, 0, 1, 2, 0]], atol=1e-12)


def test_monotonicity_check_reads_the_same_kernel() -> None:
    """The check is this curve plus a grid validation and a decrease report."""
    grid = np.linspace(-3.0, 3.0, 25)
    report = check_bifactor_expected_total_score_monotonicity(
        _anchored(), grid, q_specific=41, group=1
    )
    np.testing.assert_allclose(
        report.expected_total,
        predict_bifactor_expected_total_score(_anchored(), grid, q_specific=41),
        atol=1e-12,
    )
    assert report.monotone


def test_high_order_shared_rule_returns_a_finite_stable_curve() -> None:
    """The Rust rule remains finite beyond NumPy hermegauss's failure order."""
    grid = np.linspace(-3.0, 3.0, 7)
    high = predict_bifactor_expected_total_score(
        _anchored(), grid, q_specific=481
    )
    reference = predict_bifactor_expected_total_score(
        _anchored(), grid, q_specific=121
    )
    assert np.all(np.isfinite(high))
    np.testing.assert_allclose(high, reference, atol=1e-9, rtol=0.0)


def test_rejects_non_finite_theta() -> None:
    with pytest.raises(ValueError):
        predict_bifactor_expected_total_score(
            _anchored(), np.array([0.0, np.nan]), q_specific=41
        )


@pytest.mark.parametrize(
    "native_result",
    (
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
    ),
)
def test_rejects_invalid_native_expected_total(monkeypatch, native_result) -> None:
    """The public consumer rejects malformed or non-finite FFI results."""
    import fast_mlsirm.polytomous as polytomous

    class _Core:
        @staticmethod
        def bifactor_expected_total_score(*args):
            return native_result

    monkeypatch.setattr(polytomous, "_core_module", lambda: _Core())
    with pytest.raises(ValueError, match="wrong shape|not finite"):
        predict_bifactor_expected_total_score(
            _anchored(), np.array([-1.0, 1.0]), q_specific=41
        )


def _without_threshold() -> object:
    fit = _SingleGroupFit(A_GENERAL, A_SPECIFIC)
    del fit.threshold
    return fit


def _two_d_multigroup_threshold() -> _MultigroupFit:
    fit = _anchored()
    fit.threshold = fit.threshold[0]
    return fit


def _one_d_specific_sd() -> _MultigroupFit:
    fit = _anchored()
    fit.specific_sd = np.ones(3)
    return fit


def _non_finite_threshold() -> _SingleGroupFit:
    fit = _SingleGroupFit(A_GENERAL, A_SPECIFIC)
    fit.threshold[0, 0] = np.nan
    return fit


@pytest.mark.parametrize(
    ("make_fit", "theta", "error", "match"),
    [
        (_without_threshold, np.zeros(3), TypeError, "threshold array"),
        (_two_d_multigroup_threshold, np.zeros(3), ValueError, "n_groups x n_items"),
        (_one_d_specific_sd, np.zeros(3), ValueError, "n_groups x n_specific"),
        (_non_finite_threshold, np.zeros(3), ValueError, "threshold must be finite"),
        (_anchored, np.zeros((2, 2)), ValueError, "non-empty 1-D"),
        (_anchored, np.array([]), ValueError, "non-empty 1-D"),
    ],
)
def test_rejects_malformed_fit_or_theta(make_fit, theta, error, match) -> None:
    with pytest.raises(error, match=match):
        predict_bifactor_expected_total_score(make_fit(), theta, q_specific=41)


def test_multigroup_fit_without_specific_sd_scores() -> None:
    """A fit object that omits specific_sd is read as unit specific SDs."""
    fit = _anchored()
    del fit.specific_sd
    grid = np.linspace(-2.0, 2.0, 5)
    np.testing.assert_allclose(
        predict_bifactor_expected_total_score(fit, grid, q_specific=41, group=0),
        predict_bifactor_expected_total_score(_anchored(), grid, q_specific=41),
        atol=1e-12,
    )
