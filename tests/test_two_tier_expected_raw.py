"""Tests for two-tier expected raw total E[T|theta_focal] (manuscript G+4+W path)."""
from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.polytomous import check_bifactor_expected_total_score_monotonicity
from fast_mlsirm.two_tier_grm import expected_total_score_two_tier_given_primary


def test_q_nuisance_required_and_bounded() -> None:
    ap = np.array([[1.0, 0.0], [1.2, 0.0]])
    asp = np.array([0.5, 0.4])
    th = np.array([[1.0, 0.0, -1.0], [0.5, -0.2, -1.2]])
    smap = np.array([0, 0], dtype=np.int64)
    grid = np.linspace(-2.0, 2.0, 5)
    with pytest.raises(TypeError):
        expected_total_score_two_tier_given_primary(
            ap, asp, th, smap, grid, focal_primary=0  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError):
        expected_total_score_two_tier_given_primary(
            ap, asp, th, smap, grid, focal_primary=0, q_nuisance=0
        )


def test_matches_bifactor_monotonicity_when_no_extra_primary() -> None:
    """One primary + specifics ⇒ same curve as bifactor E[T|G] helper."""
    rng = np.random.default_rng(0)
    n_items = 6
    a_g = rng.uniform(0.6, 1.4, size=n_items)
    a_s = rng.uniform(0.3, 0.9, size=n_items)
    th = np.sort(rng.normal(size=(n_items, 3)), axis=1)[:, ::-1]
    smap = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    ap = a_g.reshape(-1, 1)
    grid = np.linspace(-3.0, 3.0, 17)
    got = expected_total_score_two_tier_given_primary(
        ap, a_s, th, smap, grid, focal_primary=0, q_nuisance=21
    )

    class _BF:
        a_general = a_g
        a_specific = a_s
        threshold = th

    ref = check_bifactor_expected_total_score_monotonicity(_BF(), grid, q_specific=21)
    assert np.allclose(got.expected_total, ref.expected_total, atol=1e-10, rtol=0.0)


def test_wording_cross_loads_change_curve_vs_specific_only() -> None:
    """Crossed W loadings must alter E[T|G] vs integrating only specifics."""
    a_g = np.array([1.0, 1.0, 1.0, 1.0])
    a_w = np.array([0.0, 0.0, 0.8, 0.8])  # last two crossed
    a_s = np.array([0.5, 0.5, 0.5, 0.5])
    th = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    smap = np.array([0, 1, 2, 3], dtype=np.int64)
    ap = np.column_stack([a_g, a_w])
    grid = np.linspace(-2.0, 2.0, 9)
    with_w = expected_total_score_two_tier_given_primary(
        ap, a_s, th, smap, grid, focal_primary=0, q_nuisance=11
    )
    no_w = expected_total_score_two_tier_given_primary(
        np.column_stack([a_g, np.zeros(4)]),
        a_s,
        th,
        smap,
        grid,
        focal_primary=0,
        q_nuisance=11,
    )
    assert not np.allclose(with_w.expected_total, no_w.expected_total)


def test_person_eap_points_accepted_without_ascending_grid() -> None:
    """Person G EAP vectors need not be sorted (unlike monotonicity helpers)."""
    ap = np.array([[1.0, 0.4], [1.1, 0.0]])
    asp = np.array([0.5, 0.6])
    th = np.array([[0.5, -0.5], [0.2, -0.8]])
    smap = np.array([0, 0], dtype=np.int64)
    eap = np.array([0.3, -1.2, 0.3, 2.0])  # unsorted / repeats OK
    out = expected_total_score_two_tier_given_primary(
        ap, asp, th, smap, eap, focal_primary=0, q_nuisance=9
    )
    assert out.expected_total.shape == eap.shape
    assert out.expected_total[0] == pytest.approx(out.expected_total[2])
