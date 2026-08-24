"""Real computerized-adaptive-test (CAT) accuracy AND efficiency test for
fast_mlsirm's cat_simulate_polytomous (Dodd, De Ayala & Koch, 1995):
fits a GRM item bank from known true item parameters and person thetas
(same simulation approach as test_polytomous_grm_recovery.py), then runs
the adaptive simulator against known true thetas and asserts BOTH that
theta recovery stays close to full-bank accuracy AND that CAT actually
uses substantially fewer items than the full bank -- the property that
distinguishes a real CAT test from just another full-bank recovery test.
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import cat_simulate_polytomous, fit_polytomous

N_PERSONS = 400
N_ITEMS = 40
N_CAT = 4
SEED = 20260101

CAT_MIN_ITEMS = 5
CAT_MAX_ITEMS = N_ITEMS
CAT_SE_THRESHOLD = 0.4

# A real run with these exact parameters/seed measures theta RMSE ~0.40 and
# correlation ~0.91 using a mean of ~8.7 of 40 items -- comparable accuracy
# to the full-bank GRM recovery test (~0.38/~0.92) at roughly a fifth of
# the items. Margins are loose enough to tolerate a minor fast-mlsirm
# version bump while still catching a real regression in either accuracy
# or the adaptive-selection efficiency CAT exists to provide.
#
# MAX_MEAN_ITEMS_USED is deliberately close to the measured ~8.7 (not a
# loose N_ITEMS * 0.5): the same fixture/seed with adaptive=False (random
# item order) measures mean_items_used ~14.97, which still clears rmse/
# correlation bounds -- so a bound of 12 is what actually catches a silent
# fallback to non-adaptive selection (e.g. an `adaptive` flag dropped on
# its way through the Rust binding).
MAX_THETA_RMSE = 0.65
MIN_THETA_CORRELATION = 0.7
MAX_MEAN_ITEMS_USED = 12


def _grm_category_probs(theta: float, discrimination: float, thresholds: np.ndarray) -> np.ndarray:
    """Samejima (1969) graded-response category probabilities."""
    cumulative = np.concatenate(([1.0], 1.0 / (1.0 + np.exp(-discrimination * (theta - thresholds))), [0.0]))
    return -np.diff(cumulative)


def test_cat_recovers_theta_using_substantially_fewer_items_than_the_full_bank() -> None:
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

    bank = fit_polytomous(responses, n_cat=N_CAT, model="grm")
    assert bank.converged

    cat_result = cat_simulate_polytomous(
        true_theta,
        bank,
        min_items=CAT_MIN_ITEMS,
        max_items=CAT_MAX_ITEMS,
        se_threshold=CAT_SE_THRESHOLD,
        adaptive=True,
        seed=SEED,
    )
    theta_eap = cat_result["theta_eap"]
    n_used = cat_result["n_used"]

    rmse = float(np.sqrt(np.mean((theta_eap - true_theta) ** 2)))
    correlation = float(np.corrcoef(theta_eap, true_theta)[0, 1])
    mean_items_used = float(n_used.mean())

    assert rmse < MAX_THETA_RMSE, f"CAT theta RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    assert correlation > MIN_THETA_CORRELATION, f"CAT theta correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
    assert mean_items_used < MAX_MEAN_ITEMS_USED, (
        f"CAT used a mean of {mean_items_used:.2f} of {N_ITEMS} items, "
        f"not meaningfully fewer than the full bank -- adaptive item selection isn't providing efficiency"
    )
