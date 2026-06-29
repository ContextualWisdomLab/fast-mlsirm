import numpy as np

from fast_mlsirm import MLS2PLMConfig, simulate
from fast_mlsirm.diagnostics import predict_proba


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
