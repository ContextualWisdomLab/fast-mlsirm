"""Theta-recovery study for the Rust-backed generalized partial credit model.

Responses are generated from known person abilities, discriminations, and step
parameters using Muraki's GPCM, then fit and scored through the public
``fit_polytomous(..., model="gpcm")`` and ``score_polytomous`` surfaces. The
simulator uses cumulative step logits
``z_k = sum_{v=1}^k a * (theta - b_v)``, ``z_0 = 0``, followed by a softmax.
That step-difficulty representation is equivalent to the additive category
intercepts documented by ``PolytomousFit``; production GPCM arithmetic remains
in Rust.

Reference
---------
Muraki, E. (1992). A generalized partial credit model: Application of an EM
algorithm. *Applied Psychological Measurement, 16*(2), 159-176.
https://doi.org/10.1177/014662169201600206
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
# at the same sample size. The margins below stay loose enough to tolerate
# a minor fast-mlsirm version bump while still catching a real recovery
# regression.
MAX_THETA_RMSE = 0.55
MIN_THETA_CORRELATION = 0.8


def _gpcm_category_probs(
    theta: float, discrimination: float, steps: np.ndarray
) -> np.ndarray:
    """Muraki (1992) GPCM probabilities for one person/item pair."""
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
            probs = _gpcm_category_probs(
                true_theta[person], true_discrimination[item], true_steps[item]
            )
            responses[person, item] = rng.choice(N_CAT, p=probs)

    fit = fit_polytomous(responses, n_cat=N_CAT, model="gpcm", max_iter=80)
    assert fit.converged

    scored = score_polytomous(responses, fit)
    theta_eap = scored["theta_eap"]

    rmse = float(np.sqrt(np.mean((theta_eap - true_theta) ** 2)))
    correlation = float(np.corrcoef(theta_eap, true_theta)[0, 1])

    assert rmse < MAX_THETA_RMSE, (
        f"theta recovery RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    )
    assert correlation > MIN_THETA_CORRELATION, (
        f"theta recovery correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
    )
