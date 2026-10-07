"""Steep specific-slope accuracy of two-tier expected raw scores (CPU, no fit).

Accuracy rule (maintainer decision, 2026-10-06): absolute error <= 1e-8
against an independent reference, at ``q_specific >= 3841``. Lower counts
are not accepted for steep specific slopes: the steep12 fixture at Q961 has
a retained error of 9.661355182544185e-08 and is asserted to stay outside
1e-8 so the known negative remains visible rather than being masked.

The reference integrates the category first moment against N(0, 1) by
composite Gauss-Legendre (48 panels on [-12, 12], degree 64, refinement
checked against degree 32). The conditioning measure follows Cai (2015,
pp. 542-544, Eqs. 14-20); the panel layout is a local verification choice.
These are explicit parameter fixtures, not calibrated participant data.

Reference:
Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. Psychometrika, 80(2), 535-559.
https://doi.org/10.1007/s11336-014-9411-3
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm as fm

TOLERANCE = 1e-8
ACCEPTED_Q = (3841, 7681)

_A_PRIMARY = np.array([[1.2, .3], [-.4, 1.0], [.8, -.6], [1.1, .2], [-.3, .9], [.6, .4]])
_THRESHOLDS = np.array([[.8, -.8], [1., -1.], [.5, -1.2], [2., -.3], [.4, -2.], [1.5, -.5]])
_THETA = np.array([[-3., -2.], [-1., 0.], [0., 1.], [1., -1.], [2., 3.], [0., 0.], [3., -3.]])
_SPECIFIC_MAP = np.array([0, 0, 1, 1, 2, -1], dtype=np.int64)
_SLOPES = {
    "ordinary": [.5, .4, .8, .6, 1., 9.],
    "signed": [1.2, -1.8, 2., -.7, -1.4, 9.],
    "steep4": [4., -3., 2.5, -4., 3., 9.],
    "steep12": [12., -9., 8., -12., 10., 9.],
    "zero-specific": [0., 0., 0., 0., 0., 9.],
}


def _fit(name):
    return SimpleNamespace(
        a_primary=_A_PRIMARY.copy(), a_specific=np.array(_SLOPES[name]),
        threshold=_THRESHOLDS.copy(), theta_p_eap=_THETA.copy(),
        n_primary=2, n_specific=3, n_cat=3,
    )


def _reference(fit, smap, *, degree, panels=48, limit=12.0):
    x, w = np.polynomial.legendre.leggauss(degree)
    edges = np.linspace(-limit, limit, panels + 1)
    half = np.diff(edges) / 2
    mid = (edges[:-1] + edges[1:]) / 2
    nodes = (mid[:, None] + half[:, None] * x).reshape(-1)
    weights = (half[:, None] * w).reshape(-1) * np.exp(-nodes**2 / 2) / math.sqrt(2 * math.pi)
    result = []
    for theta in fit.theta_p_eap:
        total = 0.0
        for i, block in enumerate(smap):
            base = math.fsum(float(a) * float(t) for a, t in zip(fit.a_primary[i], theta))
            latent = np.array([0.0]) if block == -1 else nodes
            measure = np.array([1.0]) if block == -1 else weights
            slope = 0.0 if block == -1 else fit.a_specific[i]
            z = base + slope * latent[:, None] + fit.threshold[i]
            cum = np.exp(-np.logaddexp(0.0, -z))
            categories = np.column_stack((1 - cum[:, 0], cum[:, :-1] - cum[:, 1:], cum[:, -1]))
            moments = categories @ np.arange(fit.n_cat, dtype=float)
            total += math.fsum(float(v) * float(m) for v, m in zip(moments, measure))
        result.append(total)
    return np.array(result)


@pytest.fixture(scope="module")
def references():
    out = {}
    for name in _SLOPES:
        fit = _fit(name)
        r32 = _reference(fit, _SPECIFIC_MAP, degree=32)
        r64 = _reference(fit, _SPECIFIC_MAP, degree=64)
        assert np.max(np.abs(r32 - r64)) <= 1e-12, name
        out[name] = r64
    return out


@pytest.mark.parametrize("q", ACCEPTED_Q)
@pytest.mark.parametrize("name", sorted(_SLOPES))
def test_expected_raw_within_1e8_of_reference_at_accepted_quadrature(references, name, q):
    result = fm.score_two_tier_grm_expected_raw(_fit(name), _SPECIFIC_MAP, q, device="cpu")
    assert result.used_gpu is False
    error = float(np.max(np.abs(result.expected_raw - references[name])))
    assert error <= TOLERANCE, f"{name} Q{q}: absolute error {error!r} > {TOLERANCE}"


def test_steep12_q961_remains_outside_tolerance(references):
    """The retained negative must stay visible; Q961 is not an accepted count."""
    result = fm.score_two_tier_grm_expected_raw(_fit("steep12"), _SPECIFIC_MAP, 961, device="cpu")
    error = float(np.max(np.abs(result.expected_raw - references["steep12"])))
    assert error > TOLERANCE, f"steep12 Q961 error {error!r} unexpectedly within {TOLERANCE}"
