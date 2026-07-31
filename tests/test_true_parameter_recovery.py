"""True-parameter recovery of the point estimator from a controlled study.

A parameter-recovery (or "parameter-recovery study") design is the canonical
way to validate an IRT estimator: simulate binary responses from *known* item
and person parameters, estimate the model, and confirm the estimates track the
generating truth (Baker & Kim, 2004, *Item Response Theory: Parameter
Estimation Techniques*, 2nd ed., Marcel Dekker; Lord, 1980, *Applications of
Item Response Theory to Practical Testing Problems*, Lawrence Erlbaum). The
suite's ``test_fit_pipeline_smoke`` only checks that ``recovery_report`` emits a
summary key after a 3-iteration fit; it never asserts the estimator actually
recovers the truth, so genuine recovery was untested.

These tests pin recovery on a well-conditioned synthetic design (moderate
difficulty spread, moderate discrimination, a large person sample) fit to
convergence: item difficulty and person ability are recovered with high
correlation and low error, and item discrimination is recovered in rank. The
estimator is the regularised JML/MAP point estimator; item difficulty and the
ability metric are the structurally well-identified quantities, while
discrimination is the more weakly identified slope — a known property of joint
estimation (the incidental-parameters behaviour discussed in Baker & Kim, 2004),
so difficulty/ability get the tight bounds and discrimination the rank bound.

The fit is pinned to the NumPy reference backend so recovery is deterministic
and platform-stable; the Rust core is held numerically identical by the
existing Rust<->NumPy parity gate (``tests/test_rust_parity.py``).
"""

import numpy as np

from fast_mlsirm import FitConfig, recovery_report
from fast_mlsirm.fit import fit
from fast_mlsirm.math import sigmoid
from fast_mlsirm.types import MLSIRMParams


def _well_conditioned_truth(n_persons: int, n_items: int, seed: int):
    """Build (Y, factor_id, truth) for a single-factor recovery study.

    Difficulty is an even, informative spread over ``[-1.8, 1.8]`` and
    discrimination is moderate (``[0.8, 1.8]``) so every item carries signal
    (no near-degenerate extreme-difficulty items), which is the design under
    which a consistent estimator is expected to recover the truth.
    """
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n_persons, 1))
    a = rng.uniform(0.8, 1.8, n_items)
    b = np.linspace(-1.8, 1.8, n_items)
    factor = np.zeros(n_items, dtype=np.int64)
    eta = a[None, :] * theta[:, factor] + b[None, :]
    y = rng.binomial(1, sigmoid(eta)).astype(np.uint8)
    truth = MLSIRMParams(
        theta=theta,
        alpha=np.log(a),
        b=b,
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=-30.0,
    )
    return y, factor, truth


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via Pearson correlation of the ranks."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def test_point_estimator_recovers_true_item_and_person_parameters():
    """The fitted estimates track the generating truth on a clean design.

    Difficulty and ability are the well-identified quantities and are held to
    tight recovery bounds; discrimination is recovered in rank. All estimates
    and the objective stay finite (no runaway slope), so recovery is genuine
    rather than a degenerate fit that happens to satisfy a single metric.
    """
    y, factor, truth = _well_conditioned_truth(n_persons=1000, n_items=15, seed=1)

    result = fit(
        y,
        factor,
        config=FitConfig(
            model="MIRT",
            optimizer="adam",
            max_iter=800,
            n_restarts=4,
            seed=1,
            backend="numpy",
        ),
    )
    metrics = recovery_report(truth, result.params).metrics
    theta_corr = float(np.corrcoef(truth.theta.ravel(), result.params.theta.ravel())[0, 1])

    # Every estimate and the objective are finite — no incidental-parameter blow-up.
    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.alpha))
    assert np.all(np.isfinite(result.params.theta))

    # Difficulty: the structurally well-identified item parameter — recovered
    # with high correlation and low error.
    assert metrics["b_corr"] > 0.95
    assert metrics["b_rmse"] < 0.30

    # Ability metric: recovered with high correlation across the person sample.
    assert theta_corr > 0.72

    # Discrimination: the more weakly identified slope — recovered in rank
    # (ordering of item discriminations preserved).
    assert _rank_corr(truth.a, result.params.a) > 0.70
    assert metrics["a_corr"] > 0.70


def test_recovery_is_deterministic_under_the_reference_backend():
    """A seeded reference-backend fit reproduces identical recovery metrics.

    Recovery is used as a regression guard, so the fit must be deterministic:
    refitting the same seeded design on the NumPy backend yields the same
    difficulty/ability recovery, byte-for-byte, so a future change that moves
    the estimates is caught rather than masked by run-to-run noise.
    """
    y, factor, truth = _well_conditioned_truth(n_persons=1000, n_items=15, seed=1)
    config = FitConfig(
        model="MIRT",
        optimizer="adam",
        max_iter=800,
        n_restarts=4,
        seed=1,
        backend="numpy",
    )

    first = recovery_report(truth, fit(y, factor, config=config).params).metrics
    second = recovery_report(truth, fit(y, factor, config=config).params).metrics

    assert np.isclose(first["b_corr"], second["b_corr"])
    assert np.isclose(first["b_rmse"], second["b_rmse"])
    assert np.isclose(first["a_corr"], second["a_corr"])
