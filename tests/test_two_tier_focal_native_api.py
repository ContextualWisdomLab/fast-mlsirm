"""Native bridge checks for Cai (2010), pp. 608-609 posterior/EM contracts.

These synthetic checks require a newly built extension. They must fail when
its focal exports are absent; old wheels do not provide acceptance evidence.
"""

import numpy as np

from fast_mlsirm.fitstats import _core_module


def test_fixed_bank_focal_native_score_fit_and_returned_state():
    core = _core_module()
    assert core is not None
    score = core.score_two_tier_grm_orthogonal
    fit = core.fit_two_tier_grm_focal_orthogonal
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, -1, -1, -1, -1], dtype=np.int64)
    observed = y >= 0
    pm = np.ones(4, dtype=np.bool_)
    sm = np.zeros(4, dtype=np.int64)
    ap = np.array([0.8, 1.2, 0.6, 1.1])
    asp = np.array([0.4, -0.3, 0.7, -0.2])
    threshold = np.tile([0.8, -0.6], 4)
    mu = np.array([0.3, -0.2])
    sd = np.array([1.1, 0.8])
    originals = [a.copy() for a in (ap, asp, threshold)]
    args = (y, observed, pm, sm, ap, asp, threshold, mu, sd, 3, 4, 1, 1, 3, 7, 7)
    initial = score(*args)
    for key in ("person_mean", "person_second", "person_sd"):
        assert np.asarray(initial[key]).shape == (6,)
    np.testing.assert_allclose(np.asarray(initial["person_mean"])[-2:], mu, atol=1e-12)
    np.testing.assert_allclose(np.asarray(initial["person_sd"])[-2:], sd, atol=1e-12)
    result = fit(*args, 1, 1e-14)
    assert result["n_iter"] == 1
    assert len(result["loglik_trace"]) == 2
    assert result["loglik_trace"][-1] == result["loglik"]
    assert result["final_loglik_change"] == result["loglik_trace"][1] - result["loglik_trace"][0]
    np.testing.assert_array_equal(result["initial_mean"], mu)
    np.testing.assert_array_equal(result["initial_sd"], sd)
    assert result["q_primary"] == result["q_specific"] == 7
    assert result["max_iter"] == 1
    assert result["tol"] == 1e-14
    repeated = score(y, observed, pm, sm, ap, asp, threshold,
                     np.asarray(result["latent_mean"]), np.asarray(result["latent_sd"]),
                     3, 4, 1, 1, 3, 7, 7)
    for key in ("person_mean", "person_second", "person_sd", "loglik"):
        np.testing.assert_array_equal(result[key], repeated[key])
    for actual, original in zip((ap, asp, threshold), originals):
        np.testing.assert_array_equal(actual, original)
