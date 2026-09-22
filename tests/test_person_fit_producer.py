"""Synthetic checks for the saved-fit person-fit producer."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.polytomous import (
    PolyFipcFit,
    PolytomousFit,
    compute_person_fit_polytomous,
    fit_polytomous,
    predict_category_probabilities_polytomous,
)


def test_person_fit_refuses_real_unconverged_fit():
    responses = np.tile(np.array([[0, 1, 2, 1], [2, 1, 0, 1]]), (12, 1))
    fit = fit_polytomous(responses, 3, model="grm", q_theta=121, max_iter=1, tol=1e-12)
    assert fit.converged is False
    with pytest.raises(ValueError, match="converged"):
        compute_person_fit_polytomous(responses, fit, q_theta=121, flag_threshold=-1.5)


def test_person_fit_fipc_prior_and_provenance():
    responses = np.array([[0, 1, 2, -1], [2, 2, 1, 0], [1, np.nan, 0, 2]])
    slope = np.array([1.1, 0.9, 1.2, 0.8])
    cat_params = np.array([[0.8, -0.8]] * 4)
    focal = PolyFipcFit(slope, cat_params, 0.7, 1.3, 0.0, 2, True, "converged")
    result = compute_person_fit_polytomous(responses, focal, q_theta=121, flag_threshold=-1.5)
    grm = PolytomousFit("grm", slope, cat_params, 0.0, 2, True, "converged")
    expected = compute_person_fit_polytomous(
        responses, grm, q_theta=121, prior_mean=focal.mu,
        prior_sd=focal.sigma, flag_threshold=-1.5,
    )
    for key in ("lz", "lz_star", "theta_eap", "flagged"):
        np.testing.assert_array_equal(result[key], expected[key])
    nodes, weights = np.polynomial.hermite.hermgauss(121)
    theta = focal.mu + np.sqrt(2) * focal.sigma * nodes
    probabilities = predict_category_probabilities_polytomous(grm, theta)
    likelihood = np.prod(
        [probabilities[:, item, int(responses[0, item])]
         for item in range(responses.shape[1]) if responses[0, item] >= 0],
        axis=0,
    )
    expected_eap = np.dot(theta, weights * likelihood) / np.dot(weights, likelihood)
    np.testing.assert_allclose(result["theta_eap"][0], expected_eap, atol=1e-8)
    np.testing.assert_array_equal(result["n_observed"], [3, 4, 3])
    assert result["fast_mlsirm_version"] == fast_mlsirm.__version__
    digest = hashlib.sha256(Path(result["core_path"]).read_bytes()).hexdigest()
    assert result["core_sha256"] == digest
    assert result["converged"] is True
    assert result["termination_reason"] == "converged"


def test_standard_prior_matches_existing_eap():
    responses = np.array([[0, 1, 2], [2, 1, 0]], dtype=float)
    slope = np.array([1.0, 1.1, 0.9])
    cat_params = np.array([[0.7, -0.7]] * 3)
    fit = PolytomousFit("grm", slope, cat_params, 0.0, 2, True, "converged")
    result = compute_person_fit_polytomous(responses, fit, q_theta=121, flag_threshold=-1.5)
    core = fast_mlsirm._core
    old_eap = core.score_poly_eap(
        responses.astype(np.int64).reshape(-1), 2, 3, 3, slope,
        cat_params.reshape(-1), None, "grm", 121,
    )["theta_eap"]
    np.testing.assert_array_equal(result["theta_eap"], old_eap)
    focal = PolyFipcFit(slope, cat_params, 0.0, 1.0, 0.0, 2, True, "converged")
    focal_result = compute_person_fit_polytomous(
        responses, focal, q_theta=121, flag_threshold=-1.5,
    )
    for key in ("lz", "lz_star", "theta_eap", "flagged"):
        np.testing.assert_array_equal(focal_result[key], result[key])


def test_person_fit_refuses_unconverged_fipc():
    fit = PolyFipcFit(
        np.array([1.0, 1.0]), np.array([[0.5, -0.5]] * 2),
        0.4, 1.2, 0.0, 1, False, "max_iter",
    )
    with pytest.raises(ValueError, match="converged"):
        compute_person_fit_polytomous(
            np.array([[0, 1]]), fit, q_theta=121, flag_threshold=-1.5,
        )
