"""Real theta-recovery test for the polytomous GPCM estimator: simulate
polytomous responses from known true item parameters and person
abilities using the generalized partial credit model (Muraki, 1993),
fit them with fast_mlsirm.fit_polytomous(..., model="gpcm") -- the model
option downstream model-selection steps (e.g.
fixed_item_calibration_diagnostics-driven pickers) can choose -- and
assert the recovered EAP thetas are close to the true thetas by RMSE and
correlation.

fast_mlsirm ships no polytomous-specific simulator, so the GPCM
category-probability formula is implemented directly here, matching the
library's own documented parameterization (PolytomousFit's docstring:
"GPCM additive category intercepts"): cumulative step logits
z_k = sum_{v=1}^{k} a*(theta - b_v), z_0 = 0, softmax over z.
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import fit_polytomous, score_polytomous

N_PERSONS = 400
N_ITEMS = 12
N_CAT = 4
SEED = 20260101

# A real run with these exact parameters/seed measures RMSE ~0.30 and
# correlation ~0.95 -- stronger recovery than the GRM test's ~0.38/~0.92
# at the same sample size, consistent with GPCM's additive (vs. GRM's
# cumulative) category structure being easier to identify here. The
# margins below stay loose enough to tolerate a minor fast-mlsirm version
# bump while still catching an actual estimation regression.
MAX_THETA_RMSE = 0.55
MIN_THETA_CORRELATION = 0.8


def _gpcm_category_probs(theta: float, discrimination: float, steps: np.ndarray) -> np.ndarray:
    """Muraki (1993) generalized partial credit model category
    probabilities for one person/item pair, given known true parameters."""
    cumulative_steps = np.cumsum(discrimination * (theta - steps))
    z = np.concatenate(([0.0], cumulative_steps))
    z = z - z.max()
    unnormalized = np.exp(z)
    return unnormalized / unnormalized.sum()


def test_gpcm_recovers_true_theta_within_expected_rmse() -> None:
    rng = np.random.default_rng(SEED)
    true_theta = rng.normal(0.0, 1.0, N_PERSONS)
    true_discrimination = rng.uniform(0.8, 2.0, N_ITEMS)
    true_steps = rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1))

    responses = np.zeros((N_PERSONS, N_ITEMS))
    for item in range(N_ITEMS):
        for person in range(N_PERSONS):
            probs = _gpcm_category_probs(true_theta[person], true_discrimination[item], true_steps[item])
            responses[person, item] = rng.choice(N_CAT, p=probs)

    fit = fit_polytomous(responses, n_cat=N_CAT, model="gpcm", max_iter=80)
    assert fit.converged

    scored = score_polytomous(responses, fit)
    theta_eap = scored["theta_eap"]

    rmse = float(np.sqrt(np.mean((theta_eap - true_theta) ** 2)))
    correlation = float(np.corrcoef(theta_eap, true_theta)[0, 1])

    assert rmse < MAX_THETA_RMSE, f"theta recovery RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    assert correlation > MIN_THETA_CORRELATION, f"theta recovery correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
