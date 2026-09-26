"""Contracts for 1-D poly-GRM FIPC group person scores + anchor reference moments."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.fipc_group_score import (
    _anchor_item_bank,
    _as_poly_fit,
    _score_poly_eap_gaussian_prior,
    execute_fipc_group_person_score_payload,
    poly_reference_expected_score_moments,
    score_poly_fipc_group_persons,
)
from fast_mlsirm.polytomous import fit_polytomous, score_polytomous

N_ITEMS = 8
N_ANCHOR = 5
N_CAT = 3
Q = 21
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
    fit = fit_polytomous(y_ref, n_cat=N_CAT, model="grm", q_theta=Q, max_iter=120, tol=1e-5)
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
        max_iter=120,
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


def test_q_theta_required_no_default(reference_bank) -> None:
    ref_slope, ref_cat = reference_bank
    y = _simulate(3, 40, 0.0, 1.0, TRUE_A)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    with pytest.raises(TypeError):
        score_poly_fipc_group_persons(  # type: ignore[call-arg]
            y, N_CAT, anchor, ref_slope, ref_cat, max_iter=40, tol=1e-4
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
        max_iter=120,
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


def test_reference_expected_score_moments_finite(reference_bank) -> None:
    ref_slope, ref_cat = reference_bank
    moments = poly_reference_expected_score_moments(
        ref_slope[:N_ANCHOR],
        ref_cat[:N_ANCHOR],
        mu_ref=0.0,
        sigma_ref=1.0,
        q_theta=Q,
    )
    assert moments.variance >= 0.0
    assert np.isfinite(moments.mean)
    assert moments.n_anchor_items == N_ANCHOR
    assert 0.0 <= moments.mean <= float(N_ANCHOR * (N_CAT - 1))


def test_reference_moments_negative_mu_and_nonunit_sigma(reference_bank) -> None:
    ref_slope, ref_cat = reference_bank
    slope_a, cat_a = ref_slope[:N_ANCHOR], ref_cat[:N_ANCHOR]
    low = poly_reference_expected_score_moments(
        slope_a, cat_a, mu_ref=-1.25, sigma_ref=0.7, q_theta=Q
    )
    high = poly_reference_expected_score_moments(
        slope_a, cat_a, mu_ref=1.25, sigma_ref=1.6, q_theta=Q
    )
    assert high.mean > low.mean
    assert low.mu_ref == -1.25
    assert high.sigma_ref == 1.6


def test_reference_moments_use_anchor_bank_only(reference_bank) -> None:
    """Free-item parameter changes must not affect payload reference moments."""
    ref_slope, ref_cat = reference_bank
    y = _simulate(7, 60, 0.3, 1.0, TRUE_A)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    payload = {
        "responses": y.tolist(),
        "n_cat": N_CAT,
        "anchor": anchor.tolist(),
        "anchor_slope": ref_slope.tolist(),
        "anchor_cat_params": ref_cat.tolist(),
        "q_theta": Q,
        "max_iter": 80,
        "tol": 1e-4,
        "reference_mu": -0.5,
        "reference_sigma": 1.3,
    }
    out = execute_fipc_group_person_score_payload(payload)
    ref = out["reference_expected_score"]
    assert ref["bank"] == "anchor_items_only"
    assert ref["n_anchor_items"] == N_ANCHOR

    slope_a, cat_a = _anchor_item_bank(anchor, ref_slope, ref_cat)
    direct = poly_reference_expected_score_moments(
        slope_a, cat_a, mu_ref=-0.5, sigma_ref=1.3, q_theta=Q
    )
    assert abs(float(ref["mean"]) - direct.mean) < 1e-10

    # Poison free-item rows in anchor_* arrays; anchor slice must be unchanged.
    poisoned_slope = ref_slope.copy()
    poisoned_cat = ref_cat.copy()
    poisoned_slope[N_ANCHOR:] = 9.9
    poisoned_cat[N_ANCHOR:] = 9.9
    payload2 = dict(payload)
    payload2["anchor_slope"] = poisoned_slope.tolist()
    payload2["anchor_cat_params"] = poisoned_cat.tolist()
    out2 = execute_fipc_group_person_score_payload(payload2)
    assert abs(float(out2["reference_expected_score"]["mean"]) - direct.mean) < 1e-10


def test_execute_payload_consumer_contract(reference_bank) -> None:
    ref_slope, ref_cat = reference_bank
    y = _simulate(7, 80, 0.3, 1.0, TRUE_A)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:N_ANCHOR] = True
    payload = {
        "responses": y.tolist(),
        "n_cat": N_CAT,
        "anchor": anchor.tolist(),
        "anchor_slope": ref_slope.tolist(),
        "anchor_cat_params": ref_cat.tolist(),
        "q_theta": Q,
        "max_iter": 100,
        "tol": 1e-5,
        "reference_mu": 0.0,
        "reference_sigma": 1.0,
    }
    out = execute_fipc_group_person_score_payload(payload)
    assert out["library_function"].endswith("score_poly_fipc_group_persons")
    assert out["model_scope"] == "unidimensional_poly_grm_fipc"
    assert out["n_persons"] == 80
    assert len(out["expected_raw"]) == 80
    assert "reference_expected_score" in out
    assert out["reference_expected_score"]["variance"] >= 0.0


def test_execute_payload_missing_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing required"):
        execute_fipc_group_person_score_payload({"responses": [[0, 1]]})
