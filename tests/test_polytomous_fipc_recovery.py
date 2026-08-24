"""Real Fixed-Item Parameter Calibration (FIPC, Kim 2006) test for the
two-stage fit-then-score design downstream consumers rely on (e.g.
LineageWeave's period reports: the first period free-calibrates and
persists its item bank; later periods EAP-score on those fixed
parameters, because independent refits would re-center each period at 0
and hide real movement).

Simulates a period-1 GRM item bank and fits it freely (the "first period
free-calibrates" step), then simulates a period-2 cohort of different
people from the SAME true item parameters but a deliberate mean-theta
shift (representing genuine week-over-week movement), and scores period 2
via score_polytomous(period2_responses, period1_fit) -- period 1's fit is
passed through unchanged, never re-estimated from period-2 data, which is
what "fixed" means here.

The test asserts two things a merely-passing "it runs" test would miss:
(1) FIPC-scored period-2 thetas recover the true mean shift reasonably
    well, and
(2) an *independent* free refit of period 2 (the alternative the
    docstring warns against) would have hidden that same shift by
    re-centering close to 0 -- proving the FIPC path is doing something
    a naive independent refit provably cannot.
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import fit_polytomous, score_polytomous

N_ITEMS = 15
N_CAT = 4
N_PERSONS_PERIOD_1 = 400
N_PERSONS_PERIOD_2 = 250
SEED = 20260101
PERIOD_2_TRUE_MEAN_SHIFT = 0.3

# A real run with these exact parameters/seed measures period-2 FIPC theta
# RMSE ~0.33 and correlation ~0.94, with the FIPC-estimated period-2 mean
# landing at ~0.37 against a true realized mean of ~0.43 (the shift is
# substantially recovered, not perfectly -- sampling noise on 250 people).
# An independent free refit of the same period-2 data, by contrast,
# re-centers to an estimated mean of ~0.01 -- essentially erasing the
# shift, exactly the failure mode FIPC exists to avoid.
MAX_THETA_RMSE = 0.6
MIN_THETA_CORRELATION = 0.75
MIN_FIPC_MEAN_SHIFT_DETECTED = PERIOD_2_TRUE_MEAN_SHIFT * 0.5
MAX_INDEPENDENT_REFIT_MEAN_SHIFT_DETECTED = PERIOD_2_TRUE_MEAN_SHIFT * 0.35


def _grm_category_probs(theta: float, discrimination: float, thresholds: np.ndarray) -> np.ndarray:
    """Samejima (1969) graded-response category probabilities."""
    cumulative = np.concatenate(([1.0], 1.0 / (1.0 + np.exp(-discrimination * (theta - thresholds))), [0.0]))
    return -np.diff(cumulative)


def _simulate_responses(theta: np.ndarray, discrimination: np.ndarray, thresholds: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_persons = len(theta)
    n_items = len(discrimination)
    responses = np.zeros((n_persons, n_items))
    for item in range(n_items):
        for person in range(n_persons):
            probs = _grm_category_probs(theta[person], discrimination[item], thresholds[item])
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            responses[person, item] = rng.choice(N_CAT, p=probs)
    return responses


def test_fipc_recovers_period_two_mean_shift_that_an_independent_refit_would_hide() -> None:
    param_rng = np.random.default_rng(SEED)
    true_discrimination = param_rng.uniform(0.8, 2.0, N_ITEMS)
    true_thresholds = np.sort(param_rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1)), axis=1)

    theta_period_1 = param_rng.normal(0.0, 1.0, N_PERSONS_PERIOD_1)
    theta_period_2 = param_rng.normal(PERIOD_2_TRUE_MEAN_SHIFT, 1.0, N_PERSONS_PERIOD_2)

    responses_period_1 = _simulate_responses(theta_period_1, true_discrimination, true_thresholds, SEED + 1)
    period_1_fit = fit_polytomous(responses_period_1, n_cat=N_CAT, model="grm")
    assert period_1_fit.converged

    responses_period_2 = _simulate_responses(theta_period_2, true_discrimination, true_thresholds, SEED + 2)

    # FIPC: score period 2 on period 1's fixed (not re-estimated) item bank.
    fipc_scored = score_polytomous(responses_period_2, period_1_fit)
    fipc_theta_eap = fipc_scored["theta_eap"]

    rmse = float(np.sqrt(np.mean((fipc_theta_eap - theta_period_2) ** 2)))
    correlation = float(np.corrcoef(fipc_theta_eap, theta_period_2)[0, 1])
    fipc_detected_shift = float(fipc_theta_eap.mean())

    assert rmse < MAX_THETA_RMSE, f"FIPC period-2 theta RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    assert correlation > MIN_THETA_CORRELATION, f"FIPC period-2 theta correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
    assert fipc_detected_shift > MIN_FIPC_MEAN_SHIFT_DETECTED, (
        f"FIPC-detected period-2 mean shift {fipc_detected_shift:.3f} did not clear "
        f"{MIN_FIPC_MEAN_SHIFT_DETECTED:.3f} -- FIPC should preserve real week-over-week movement"
    )

    # The comparison that proves FIPC matters: an independent free refit of
    # the SAME period-2 data, never told about period 1's fixed bank.
    independent_fit = fit_polytomous(responses_period_2, n_cat=N_CAT, model="grm")
    independent_scored = score_polytomous(responses_period_2, independent_fit)
    independent_detected_shift = float(independent_scored["theta_eap"].mean())

    assert abs(independent_detected_shift) < MAX_INDEPENDENT_REFIT_MEAN_SHIFT_DETECTED, (
        f"independent refit detected shift {independent_detected_shift:.3f}, expected it to "
        f"re-center near 0 (below {MAX_INDEPENDENT_REFIT_MEAN_SHIFT_DETECTED:.3f}) -- if it also "
        f"recovers the shift, this test no longer demonstrates FIPC's actual value over a naive refit"
    )
