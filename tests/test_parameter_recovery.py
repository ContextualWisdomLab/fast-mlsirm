"""Literature-grounded parameter-recovery and estimator-stability tests.

These tests exercise the public estimation surface (``simulate`` ->
``fit`` -> ``recovery_report``) end to end. They add coverage that the
existing suite does not have: ``test_irt_stability.py`` only checks the
*identity* recovery ``recovery_report(truth, truth)`` and the objective on
tiny fixtures, so an actual "generate from known parameters, re-estimate,
compare" recovery study and a run-to-run estimator-stability check were
missing. Nothing here changes any model formula, objective, gradient, or
Rust/NumPy numeric path (AGENTS.md "Formula Scope"); the tests only assert
properties of the existing estimator.

The recovery design is the standard Monte-Carlo IRT recovery study: generate
binary responses from known item/person parameters, re-estimate, and report
bias and RMSE between true and estimated parameters (Harwell, Stone, Hsu, &
Kirisci, 1996; Reckase, 2009). Because joint MLE of item parameters is
inconsistent under the incidental-parameters problem, the reliable and honest
recovery estimator here is marginal maximum likelihood (the latent trait is
integrated out over its population distribution by Gauss-Hermite quadrature and
maximised by EM), which is consistent for the item parameters (Bock & Aitkin,
1981). Marginal ML is also the estimation principle of full-information item
factor analysis, whose identified solution is unique up to the fixed latent
metric, so repeated fits are stable (Bock, Gibbons, & Muraki, 1988).

Data are generated with the latent-space interaction disabled (``gamma=0``), so
the generating model is an ordinary two-parameter logistic (2PL) item model and
the unidimensional marginal 2PL estimator (``model="ULS2PLM"``,
``estimator="mmle"``) is correctly specified for it. Acceptance targets are not
fitted to the observed seed.  Item RMSE must beat the least-squares-optimal
constant predictor (the generating-parameter mean), and standardized-theta RMSE
must beat the zero predictor whose RMSE is one by construction.  These are
model-based skill comparisons rather than empirical score thresholds.

References
----------
Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
    item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
    443-459. https://doi.org/10.1007/BF02293801
Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor
    analysis. *Applied Psychological Measurement, 12*(3), 261-280.
    https://doi.org/10.1177/014662168801200305
Harwell, M., Stone, C. A., Hsu, T.-C., & Kirisci, L. (1996). Monte Carlo
    studies in item response theory. *Applied Psychological Measurement,
    20*(2), 101-125. https://doi.org/10.1177/014662169602000201
Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
    https://doi.org/10.1007/978-0-387-89976-3
"""

from __future__ import annotations

import numpy as np

from fast_mlsirm import FitConfig, MLS2PLMConfig, fit, recovery_report, simulate


def _constant_null_rmse(values: np.ndarray) -> float:
    """Return RMSE of the least-squares-optimal constant prediction."""
    numeric_values = np.asarray(values, dtype=np.float64)
    centered_values = numeric_values - numeric_values.mean()
    return float(np.sqrt(np.mean(centered_values * centered_values)))


def _zero_null_abs_bias(values: np.ndarray) -> float:
    """Return absolute bias of a zero-parameter prediction."""
    return float(abs(np.asarray(values, dtype=np.float64).mean()))


def test_marginal_estimator_recovers_generating_2pl_item_parameters():
    """Recover known parameters with positive null-model RMSE skill.

    Grounded in the Monte-Carlo recovery-study design (Harwell et al., 1996;
    Reckase, 2009) and the consistency of marginal ML for item parameters
    (Bock & Aitkin, 1981).  The CI sentinel has one predeclared data-generating
    condition and one attempted fit; convergence therefore means zero failures
    out of one.  It is a bounded regression sentinel, not a population-wide
    Monte-Carlo claim.
    """
    # 2PL-generated data (latent-space interaction off) so the unidimensional
    # marginal 2PL estimator is correctly specified for the generating model.
    data = simulate(
        MLS2PLMConfig(
            n_persons=600, n_dims=1, items_per_dim=20, gamma=0.0, seed=40404
        )
    )

    result = fit(
        data.Y.astype(float),
        data.factor_id,
        FitConfig(
            model="ULS2PLM",
            estimator="mmle",
            max_iter=500,
            tolerance=1e-6,
            q_theta=121,
        ),
    )
    study_counts = {
        "attempted_fits": 1,
        "failed_fits": int(result.convergence_status != "converged"),
    }
    assert study_counts == {"attempted_fits": 1, "failed_fits": 0}

    report = recovery_report(data.truth, result.params)

    assert report.metrics["a_rmse"] < _constant_null_rmse(data.truth.a)
    assert report.metrics["b_rmse"] < _constant_null_rmse(data.truth.b)
    assert report.metrics["theta_rmse_standardized"] < 1.0
    assert abs(report.metrics["a_bias"]) < _zero_null_abs_bias(data.truth.a)
    assert abs(report.metrics["b_bias"]) < _zero_null_abs_bias(data.truth.b)
    # Every recovered parameter is finite.
    assert np.all(np.isfinite(result.params.a))
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.theta))


def test_marginal_item_factor_selected_outputs_are_exactly_reproducible():
    """Repeated fits return exactly equal selected public outputs.

    This is an operational determinism contract for item parameters, person
    scores, and the likelihood trace under one fixed data/configuration
    manifest. It makes no byte-level claim and uses no tolerance selected after
    observing a seed.
    """
    data = simulate(
        MLS2PLMConfig(
            n_persons=500, n_dims=1, items_per_dim=16, gamma=0.0, seed=555
        )
    )
    y = data.Y.astype(float)

    config = FitConfig(
        model="ULS2PLM",
        estimator="mmle",
        max_iter=300,
        tolerance=1e-6,
        q_theta=121,
    )
    fit_a = fit(y, data.factor_id, config)
    fit_b = fit(y, data.factor_id, config)

    assert fit_a.convergence_status == "converged"
    assert fit_b.convergence_status == "converged"

    selected_outputs = (
        (fit_a.params.a, fit_b.params.a),
        (fit_a.params.b, fit_b.params.b),
        (fit_a.params.theta, fit_b.params.theta),
        (fit_a.loglik_trace, fit_b.loglik_trace),
    )
    for first_output, second_output in selected_outputs:
        assert np.all(np.isfinite(first_output))
        assert np.all(np.isfinite(second_output))
        np.testing.assert_array_equal(first_output, second_output)
