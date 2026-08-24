"""Real theta-recovery test for the polytomous GRM estimator: simulate
polytomous responses from known true item parameters and person
abilities, fit them with fast_mlsirm.fit_polytomous (the entry point
downstream consumers such as LineageWeave's period reports call), and
assert the recovered EAP thetas are close to the true thetas by RMSE and
correlation -- not a placeholder or an infra-only smoke test.

fast_mlsirm ships no polytomous-specific simulator (only MLS2PLMConfig's
multi-level simulate()), so the GRM response-generation formula is
implemented directly here: cumulative-logistic category boundaries
(Samejima, 1969), sampled per person/item from the resulting category
probabilities.
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import fit_polytomous, score_polytomous, validate_irt_response_matrix

N_PERSONS = 400
N_ITEMS = 12
N_CAT = 4
SEED = 20260101

# Reasonable bounds for a 12-item, 4-category GRM test at this sample size:
# a real run with these exact parameters/seed measures RMSE ~0.38 and
# correlation ~0.92, comfortably inside literature-typical recovery for a
# test this length. The margins below are loose enough to tolerate a minor
# fast-mlsirm version bump while still catching an actual estimation
# regression (e.g. RMSE blowing up past ~1 std or correlation collapsing).
MAX_THETA_RMSE = 0.6
MIN_THETA_CORRELATION = 0.75


def _grm_category_probs(theta: float, discrimination: float, thresholds: np.ndarray) -> np.ndarray:
    """Samejima (1969) graded-response category probabilities for one
    person/item pair, given known true parameters."""
    cumulative = np.concatenate(([1.0], 1.0 / (1.0 + np.exp(-(discrimination * theta - thresholds))), [0.0]))
    return -np.diff(cumulative)


def test_grm_recovers_true_theta_within_expected_rmse() -> None:
    rng = np.random.default_rng(SEED)
    true_theta = rng.normal(0.0, 1.0, N_PERSONS)
    true_discrimination = rng.uniform(0.8, 2.0, N_ITEMS)
    true_thresholds = np.sort(rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1)), axis=1)

    responses = np.zeros((N_PERSONS, N_ITEMS))
    for item in range(N_ITEMS):
        for person in range(N_PERSONS):
            probs = _grm_category_probs(true_theta[person], true_discrimination[item], true_thresholds[item])
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            responses[person, item] = rng.choice(N_CAT, p=probs)

    responses = validate_irt_response_matrix(responses, item_type="polytomous", n_categories=N_CAT)
    fit = fit_polytomous(responses, n_cat=N_CAT, model="grm", max_iter=80)
    assert fit.converged

    scored = score_polytomous(responses, fit)
    theta_eap = scored["theta_eap"]

    rmse = float(np.sqrt(np.mean((theta_eap - true_theta) ** 2)))
    correlation = float(np.corrcoef(theta_eap, true_theta)[0, 1])

    assert rmse < MAX_THETA_RMSE, f"theta recovery RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    assert correlation > MIN_THETA_CORRELATION, f"theta recovery correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
