import numpy as np

from fast_mlsirm import FitConfig, MLS2PLMConfig, simulate
from fast_mlsirm.diagnostics import (
    dimensionality_diagnostics,
    fit_diagnostics,
    predict_proba,
    response_process_dimensionality_diagnostics,
    response_process_fit_diagnostics,
)


def test_predict_proba_matches_simulation():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, seed=42))

    probs = predict_proba(data.truth, data.factor_id)
    assert np.allclose(probs, data.probabilities)


def test_predict_proba_subset_persons():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, seed=42))

    sub_persons = np.array([0, 2, 4])
    probs = predict_proba(data.truth, data.factor_id, persons=sub_persons)
    assert np.allclose(probs, data.probabilities[sub_persons, :])


def test_predict_proba_subset_items():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=4, seed=42))

    sub_items = np.array([0, 3, 5])
    probs = predict_proba(data.truth, data.factor_id, items=sub_items)
    assert np.allclose(probs, data.probabilities[:, sub_items])


def test_predict_proba_subset_both():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=4, seed=42))

    sub_persons = np.array([1, 5, 8])
    sub_items = np.array([2, 6, 7])
    probs = predict_proba(data.truth, data.factor_id, persons=sub_persons, items=sub_items)
    assert np.allclose(probs, data.probabilities[np.ix_(sub_persons, sub_items)])

import pytest
from fast_mlsirm.diagnostics import align_latent_space
from fast_mlsirm.types import MLSIRMParams

def test_align_latent_space_invalid_method():
    with pytest.raises(ValueError, match="only procrustes alignment is supported"):
        align_latent_space(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), method="invalid")

def test_predict_proba_no_space():
    truth = MLSIRMParams(theta=np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=1.0)
    probs = predict_proba(truth, np.zeros(2, dtype=int), model="MIRT")
    assert probs is not None


def test_fit_diagnostics_balanced_mirt_contract():
    params = MLSIRMParams(
        theta=np.zeros((2, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 2)),
        zeta=np.zeros((2, 2)),
        tau=0.0,
    )
    responses = np.array([[1.0, 0.0], [0.0, 1.0]])

    diagnostics = fit_diagnostics(responses, params, np.zeros(2, dtype=int), model="MIRT")

    assert np.allclose(diagnostics.itemfit["observed_count"], [2.0, 2.0])
    assert np.allclose(diagnostics.itemfit["infit_mnsq"], [1.0, 1.0])
    assert np.allclose(diagnostics.itemfit["outfit_mnsq"], [1.0, 1.0])
    assert np.allclose(diagnostics.personfit["infit_mnsq"], [1.0, 1.0])
    assert np.allclose(diagnostics.factorfit["observed_count"], [4.0])
    assert np.isclose(diagnostics.model_fit["loglik"], 4 * np.log(0.5))
    assert diagnostics.model_fit["parameter_count"] == 6.0


def test_fit_diagnostics_strata_contract():
    params = MLSIRMParams(
        theta=np.zeros((4, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((4, 2)),
        zeta=np.zeros((2, 2)),
        tau=0.0,
    )
    responses = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    diagnostics = fit_diagnostics(
        responses,
        params,
        np.zeros(2, dtype=int),
        model="MIRT",
        group_id=np.array([0, 0, 1, 1]),
        cluster_id=np.array([10, 10, 20, 20]),
    )

    assert np.allclose(diagnostics.personfit["group_id"], [0.0, 0.0, 1.0, 1.0])
    assert np.allclose(diagnostics.groupfit["observed_count"], [4.0, 4.0])
    assert diagnostics.group_itemfit["item_id"].shape == (4,)
    assert np.allclose(diagnostics.clusterfit["cluster_id"], [10.0, 20.0])


def test_dimensionality_diagnostics_returns_best_candidate():
    data = simulate(MLS2PLMConfig(n_persons=12, n_dims=2, items_per_dim=3, latent_dim=2, seed=7))

    report = dimensionality_diagnostics(
        data.Y,
        data.factor_id,
        latent_dims=[1, 2],
        k_folds=2,
        seed=5,
        config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=1, n_restarts=1, seed=5),
    )

    assert [row["latent_dim"] for row in report.candidates] == [1.0, 2.0]
    assert report.best in report.candidates
    assert all(np.isfinite(row["heldout_loglik"]) for row in report.candidates)


def test_response_process_fit_diagnostics_polytomous_contract():
    responses = np.array([[0, 1], [2, 1]])
    probabilities = np.full((2, 2, 3), 1.0 / 3.0)

    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="polytomous",
        response_process="cumulative",
    )

    assert np.isclose(diagnostics.model_fit["loglik"], 4 * np.log(1.0 / 3.0))
    assert np.allclose(diagnostics.itemfit["observed_count"], [2.0, 2.0])
    assert np.allclose(diagnostics.personfit["observed_count"], [2.0, 2.0])
    assert diagnostics.categoryfit["item_id"].shape == (6,)


def test_response_process_fit_diagnostics_dichotomous_matrix():
    responses = np.array([[1, 0], [0, 1]])
    probabilities = np.full((2, 2), 0.5)

    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="dichotomous",
        response_process="ideal_point",
    )

    assert np.isclose(diagnostics.model_fit["loglik"], 4 * np.log(0.5))
    assert diagnostics.categoryfit["category_id"].shape == (4,)


def test_response_process_fit_diagnostics_strata_contract():
    responses = np.array([[0, 1], [2, 1], [1, 0], [0, 2]])
    probabilities = np.full((4, 2, 3), 1.0 / 3.0)

    diagnostics = response_process_fit_diagnostics(
        responses,
        probabilities,
        item_type="polytomous",
        response_process="cumulative",
        group_id=np.array([0, 0, 1, 1]),
        cluster_id=np.array([10, 10, 20, 20]),
    )

    assert np.allclose(diagnostics.groupfit["observed_count"], [4.0, 4.0])
    assert diagnostics.group_itemfit["item_id"].shape == (4,)
    assert np.allclose(diagnostics.clusterfit["cluster_id"], [10.0, 20.0])


def test_response_process_dimensionality_diagnostics_selects_best_candidate():
    responses = np.array([[0, 1], [1, 0]])
    weak = np.full((2, 2, 2), 0.5)
    strong = np.array([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])

    diagnostics = response_process_dimensionality_diagnostics(
        responses,
        {"dim1": weak, "dim2": strong},
        item_type="dichotomous",
        response_process="ideal_point",
    )

    assert diagnostics.best["candidate_label"] == "dim2"
    assert [row["candidate_label"] for row in diagnostics.candidates] == ["dim1", "dim2"]
