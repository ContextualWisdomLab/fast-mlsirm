"""Contracts for 1-D poly-GRM FIPC group person scores + anchor reference moments."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fipc_group_score as fipc_group_score

from fast_mlsirm.fipc_group_score import (
    _as_poly_fit,
    _score_poly_eap_gaussian_prior,
    score_poly_fipc_group_persons,
)
from fast_mlsirm.polytomous import fit_polytomous, score_polytomous

N_ITEMS = 8
N_ANCHOR = 5
N_CAT = 3
Q = 121
TRUE_A = np.array([1.4, 1.1, 0.9, 1.2, 1.0, 0.8, 1.3, 1.15])
TRUE_D = np.array(
    [
        [1.2, -0.2],
        [1.0, 0.1],
        [1.3, -0.4],
        [0.9, 0.0],
        [1.1, -0.3],
        [1.4, -0.1],
        [1.0, -0.2],
        [1.2, 0.2],
    ]
)


def _simulate(seed: int, n_persons: int, mean: float, sd: float, slopes: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(mean, sd, n_persons)
    y = np.zeros((n_persons, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        p1 = 1.0 / (1.0 + np.exp(-(slopes[i] * theta + TRUE_D[i, 0])))
        p2 = 1.0 / (1.0 + np.exp(-(slopes[i] * theta + TRUE_D[i, 1])))
        u = rng.uniform(0.0, 1.0, n_persons)
        y[:, i] = (u < p1).astype(np.int64) + (u < p2).astype(np.int64)
    return y


@pytest.fixture(scope="module")
def reference_bank() -> tuple[np.ndarray, np.ndarray]:
    y_ref = _simulate(11, 200, 0.0, 1.0, TRUE_A)
    fit = fit_polytomous(y_ref, n_cat=N_CAT, model="grm", q_theta=Q, max_iter=500, tol=1e-5)
    assert fit.converged
    return np.asarray(fit.slope), np.asarray(fit.cat_params)


def test_score_poly_fipc_group_persons_smoke(reference_bank) -> None:
    ref_slope, ref_cat = reference_bank
    y_foc = _simulate(22, 120, 0.4, 1.1, TRUE_A)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    out = score_poly_fipc_group_persons(
        y_foc,
        N_CAT,
        anchor,
        ref_slope,
        ref_cat,
        q_theta=Q,
        max_iter=500,
        tol=1e-5,
    )
    assert out.theta_eap.shape == (120,)
    assert out.expected_raw.shape == (120,)
    assert np.all(np.isfinite(out.theta_eap))
    assert np.all(np.isfinite(out.expected_raw))
    hi = float(N_ITEMS * (N_CAT - 1))
    assert float(out.expected_raw.min()) >= -1e-9
    assert float(out.expected_raw.max()) <= hi + 1e-9
    np.testing.assert_array_equal(out.fipc.slope[:N_ANCHOR], ref_slope[:N_ANCHOR])


def test_nonconverged_fipc_never_returns_person_scores(monkeypatch) -> None:
    class UnconvergedFit:
        converged = False
        termination_reason = "max_iter"
        n_iter = 1

    monkeypatch.setattr(fipc_group_score, "fit_poly_fipc", lambda *args, **kwargs: UnconvergedFit())
    monkeypatch.setattr(
        fipc_group_score,
        "_score_poly_eap_gaussian_prior",
        lambda *args, **kwargs: pytest.fail("nonconverged bank reached EAP scoring"),
    )
    with pytest.raises(RuntimeError, match="FIPC calibration did not converge: termination_reason=max_iter, n_iter=1"):
        score_poly_fipc_group_persons(
            np.array([[0, 1]], dtype=np.int64),
            3,
            np.array([True, False]),
            np.array([1.0, 1.0]),
            np.array([[0.5, -0.5], [0.5, -0.5]]),
            q_theta=121,
            max_iter=1,
            tol=1e-5,
        )


def test_q_theta_required_no_default(reference_bank) -> None:
    ref_slope, ref_cat = reference_bank
    y = _simulate(3, 40, 0.0, 1.0, TRUE_A)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    with pytest.raises(TypeError):
        score_poly_fipc_group_persons(  # type: ignore[call-arg]
            y, N_CAT, anchor, ref_slope, ref_cat, max_iter=500, tol=1e-4
        )


def test_eap_uses_focal_prior_not_dropped_n01(reference_bank) -> None:
    """Fitted focal prior must enter EAP; score_polytomous N(0,1) is not a substitute."""
    ref_slope, ref_cat = reference_bank
    y_foc = _simulate(22, 100, 0.8, 1.2, TRUE_A)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    out = score_poly_fipc_group_persons(
        y_foc,
        N_CAT,
        anchor,
        ref_slope,
        ref_cat,
        q_theta=Q,
        max_iter=500,
        tol=1e-5,
    )
    bank = _as_poly_fit(out.fipc)
    dropped = score_polytomous(y_foc, bank, q_theta=Q)
    # Focal mean should be away from 0 for this fixture; EAP must differ from N(0,1).
    assert abs(float(out.fipc.mu)) > 0.15 or abs(float(out.fipc.sigma) - 1.0) > 0.05
    assert float(np.max(np.abs(out.theta_eap - dropped["theta_eap"]))) > 1e-3

    # Direct prior injection: non-unit negative prior shifts EAP vs N(0,1).
    neg = _score_poly_eap_gaussian_prior(
        y_foc, bank, mu=-0.75, sigma=1.4, q_theta=Q
    )
    unit = _score_poly_eap_gaussian_prior(
        y_foc, bank, mu=0.0, sigma=1.0, q_theta=Q
    )
    assert float(np.mean(neg["theta_eap"])) < float(np.mean(unit["theta_eap"]))




def test_prior_matches_brute_force_and_standard_score(reference_bank) -> None:
    slope, cat = reference_bank
    from fast_mlsirm.polytomous import PolytomousFit, predict_category_probabilities_polytomous
    bank = PolytomousFit("grm", slope, cat, 0.0, 0)
    y = np.array([[0] * N_ITEMS, [2] * N_ITEMS, [np.nan] + [1] * (N_ITEMS - 1), [-1] * N_ITEMS])
    focal = _score_poly_eap_gaussian_prior(y, bank, mu=0.6, sigma=1.3, q_theta=481)
    grid = np.linspace(-8, 8, 40001)
    probs = predict_category_probabilities_polytomous(bank, grid)
    prior = np.exp(-0.5 * ((grid - 0.6) / 1.3) ** 2)
    for p, row in enumerate(y):
        posterior = prior.copy()
        for i, value in enumerate(row):
            if np.isfinite(value) and value >= 0:
                posterior *= probs[:, i, int(value)]
        posterior /= posterior.sum()
        mean = np.sum(posterior * grid)
        sd = np.sqrt(np.sum(posterior * (grid - mean) ** 2))
        np.testing.assert_allclose([focal["theta_eap"][p], focal["theta_sd"][p]], [mean, sd], atol=1e-6)
    unit = _score_poly_eap_gaussian_prior(y, bank, mu=0, sigma=1, q_theta=Q)
    native = score_polytomous(y, bank, q_theta=Q)
    np.testing.assert_allclose(unit["theta_eap"], native["theta_eap"], atol=1e-10)
    np.testing.assert_allclose(unit["theta_sd"], native["theta_sd"], atol=1e-10)
    assert np.isfinite(focal["theta_eap"]).all()
    assert np.isfinite(focal["theta_sd"]).all()
    assert abs(focal["theta_eap"][-1] - 0.6) < 1e-10


@pytest.mark.parametrize("bad", [np.inf, 0.5, -2, 3])
def test_invalid_responses(reference_bank, bad) -> None:
    from fast_mlsirm.polytomous import PolytomousFit
    slope, cat = reference_bank
    bank = PolytomousFit("grm", slope, cat, 0.0, 0)
    y = np.zeros((1, N_ITEMS))
    y[0, 0] = bad
    with pytest.raises(ValueError):
        _score_poly_eap_gaussian_prior(y, bank, mu=0, sigma=1, q_theta=Q)
