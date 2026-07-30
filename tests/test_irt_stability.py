import numpy as np

from fast_mlsirm import FitConfig, MLS2PLMConfig, MLSIRMParams, PenaltyConfig, recovery_report, score_wle, simulate
from fast_mlsirm.diagnostics import fit_diagnostics, predict_proba
from fast_mlsirm.fit import fit
from fast_mlsirm.inference import observed_information, second_order_test, standard_errors_from_vcov, vcov_from_hessian
from fast_mlsirm.linking import link_fixed_item_parameters
from fast_mlsirm.objective import neg_loglik_and_grad, prepare_response
from fast_mlsirm.test_design import assemble_test_form, item_information, select_cat_item


def _all_finite(table: dict[str, np.ndarray]) -> bool:
    return all(np.all(np.isfinite(value)) for value in table.values())


def test_prepare_response_keeps_missing_by_design_axes():
    responses = np.array([[1.0, -1.0], [np.nan, np.nan], [0.0, -1.0]])

    clean, observed = prepare_response(responses)

    assert observed.sum(axis=1).tolist() == [1, 0, 1]
    assert observed.sum(axis=0).tolist() == [2, 0]
    assert np.array_equal(clean[1], np.array([0.0, 0.0]))


def test_objective_and_diagnostics_are_finite_with_all_missing_axes():
    params = MLSIRMParams(
        theta=np.array([[-0.5], [0.0], [0.5]]),
        alpha=np.array([0.0, 0.1]),
        b=np.array([0.0, 0.2]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[1.0, -1.0], [np.nan, np.nan], [0.0, -1.0]])
    factors = np.zeros(2, dtype=int)

    objective, grad, _ = neg_loglik_and_grad(responses, factors, params, FitConfig(model="MIRT", max_iter=1))
    diagnostics = fit_diagnostics(responses, params, factors, model="MIRT")

    assert np.isfinite(objective)
    assert np.all(np.isfinite(grad.theta))
    assert np.allclose(diagnostics.itemfit["observed_count"], [2.0, 0.0])
    assert np.allclose(diagnostics.personfit["observed_count"], [1.0, 0.0, 1.0])
    assert _all_finite(diagnostics.itemfit)
    assert _all_finite(diagnostics.personfit)
    assert _all_finite(diagnostics.factorfit)


def test_fit_handles_missing_by_design_axes_and_extreme_scores():
    responses = np.array(
        [
            [1.0, -1.0, 1.0],
            [0.0, -1.0, 0.0],
            [np.nan, np.nan, np.nan],
            [1.0, -1.0, 1.0],
        ]
    )

    result = fit(
        responses,
        np.zeros(3, dtype=int),
        config=FitConfig(model="MIRT", optimizer="adam", max_iter=1, n_restarts=1, latent_dim=1, seed=5),
    )

    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.theta))


def test_true_parameters_reproduce_simulation_probabilities():
    data = simulate(MLS2PLMConfig(n_persons=8, n_dims=2, items_per_dim=2, latent_dim=2, seed=9))

    probabilities = predict_proba(data.truth, data.factor_id)
    report = recovery_report(data.truth, data.truth.copy())

    assert np.allclose(probabilities, data.probabilities)
    assert report.summary["distance_rmse"] < 1e-12
    assert report.summary["gamma_abs_error"] == 0.0


def test_hessian_vcov_standard_errors_and_second_order_check_are_stable():
    params = MLSIRMParams(
        theta=np.array([[-0.6], [0.2], [0.8]]),
        alpha=np.array([0.1, -0.2]),
        b=np.array([0.0, 0.4]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    config = FitConfig(
        model="MIRT",
        max_iter=1,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
        rust_device="auto",
    )

    hessian = observed_information(responses, np.zeros(2, dtype=int), params, config=config, step=1e-4)
    check = second_order_test(hessian)
    vcov = vcov_from_hessian(hessian)
    standard_errors = standard_errors_from_vcov(vcov)

    assert hessian.shape == (7, 7)
    assert check["passed"] is True
    assert check["min_eigenvalue"] > 0.0
    assert np.all(np.isfinite(vcov))
    assert np.all(standard_errors > 0.0)
    assert second_order_test(np.diag([1.0, -1.0]))["passed"] is False


def test_observed_information_se_chain_is_preserved_at_a_converged_mle():
    """The SE=TRUE observed-information chain is self-consistent at a fitted MLE.

    Efron & Hinkley (1978) justify using the observed information ``I(x) = -H``
    (the negative Hessian of the log-likelihood at the estimate) for the
    asymptotic covariance of the maximum-likelihood estimator. This exercises
    that contract end to end on a *converged* model rather than an arbitrary
    point: the observed information is positive definite (the estimate is a
    genuine local maximum, so the curvature-based SEs are defined),
    ``vcov_from_hessian`` truly inverts it (``vcov @ H == I`` to numerical
    precision — the preservation property), and the reported standard errors
    are finite, positive, and equal to ``sqrt(diag(vcov))``.

    References
    ----------
    Efron, B., & Hinkley, D. V. (1978). Assessing the accuracy of the maximum
    likelihood estimator: Observed versus expected Fisher information.
    *Biometrika, 65*(3), 457-487. https://doi.org/10.1093/biomet/65.3.457
    """
    data = simulate(
        MLS2PLMConfig(n_persons=6, n_dims=1, items_per_dim=2, latent_dim=1, seed=3)
    )
    responses = np.asarray(data.Y, dtype=float)
    config = FitConfig(
        model="MIRT",
        optimizer="adam",
        max_iter=400,
        n_restarts=1,
        latent_dim=1,
        seed=3,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
    )
    result = fit(responses, data.factor_id, config=config)
    assert result.convergence_status == "converged"

    hessian = observed_information(
        responses, data.factor_id, result.params, config=config, step=1e-4
    )
    check = second_order_test(hessian)
    vcov = vcov_from_hessian(hessian)
    standard_errors = standard_errors_from_vcov(vcov)

    # Positive-definite observed information at the estimate => genuine local max.
    assert check["passed"] is True
    assert check["min_eigenvalue"] > 0.0
    # vcov is the true inverse of the observed information (SE preservation).
    assert np.allclose(vcov @ hessian, np.eye(hessian.shape[0]), atol=1e-8)
    # Reported SEs are finite, positive, and consistent with the vcov diagonal.
    assert np.all(np.isfinite(standard_errors))
    assert np.all(standard_errors > 0.0)
    assert np.allclose(np.diag(vcov), standard_errors**2)


def test_observed_information_defaults_rust_hessian_to_cpu_device(monkeypatch):
    calls = []
    params = MLSIRMParams(
        theta=np.array([[-0.2], [0.4]]),
        alpha=np.array([0.1, -0.3]),
        b=np.array([0.2, 0.5]),
        xi=np.zeros((2, 1)),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    responses = np.array([[0.0, 1.0], [1.0, 0.0]])
    factors = np.zeros(2, dtype=int)

    def fake_neg_loglik_and_grad(
        responses,
        factor_id,
        candidate,
        config=None,
        mask=None,
        backend="numpy",
        device=None,
    ):
        calls.append((backend, device))
        value = (
            np.vdot(candidate.theta, candidate.theta)
            + np.vdot(candidate.alpha, candidate.alpha)
            + np.vdot(candidate.b, candidate.b)
        )
        return float(value), candidate.copy(), -float(value)

    monkeypatch.setattr(
        "fast_mlsirm.inference.neg_loglik_and_grad", fake_neg_loglik_and_grad
    )

    hessian = observed_information(
        responses,
        factors,
        params,
        config=FitConfig(model="MIRT", backend="rust", rust_device="auto"),
        step=1e-4,
    )

    assert hessian.shape == (6, 6)
    assert calls
    assert {backend for backend, _ in calls} == {"rust"}
    assert {device for _, device in calls} == {"cpu"}


def test_fixed_item_parameter_linking_recovers_anchor_metric():
    source = MLSIRMParams(
        theta=np.array([[-1.0], [0.0], [1.0]]),
        alpha=np.log(np.array([1.2, 0.8, 1.5])),
        b=np.array([-0.4, 0.1, 0.7]),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((3, 1)),
        tau=-30.0,
    )
    scale = 1.5
    shift = -0.25
    target = source.copy()
    target.theta = scale * source.theta + shift
    target.alpha = np.log(source.a / scale)
    target.b = source.b - target.a * shift

    linked, transform = link_fixed_item_parameters(source, target, anchor_items=np.array([0, 1, 2]))

    assert np.isclose(transform["scale"][0], scale)
    assert np.isclose(transform["shift"][0], shift)
    assert np.allclose(linked.theta, target.theta)
    assert np.allclose(linked.alpha, target.alpha)
    assert np.allclose(linked.b, target.b)


def test_cat_item_selection_and_greedy_ata_constraints():
    params = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.log(np.array([0.5, 2.0, 1.0, 1.5])),
        b=np.zeros(4),
        xi=np.zeros((1, 1)),
        zeta=np.zeros((4, 1)),
        tau=-30.0,
    )
    factors = np.zeros(4, dtype=int)

    information = item_information(params, factors, theta=np.array([0.0]), model="MIRT")
    next_item = select_cat_item(params, factors, theta=np.array([0.0]), administered=np.array([1]), model="MIRT")
    form = assemble_test_form(
        information,
        length=3,
        content=np.array(["algebra", "algebra", "geometry", "geometry"]),
        min_per_content={"geometry": 1},
        max_per_content={"algebra": 1},
    )

    assert int(np.argmax(information)) == 1
    assert next_item == 3
    assert len(form) == 3
    assert np.sum(np.array(["algebra", "algebra", "geometry", "geometry"])[form] == "algebra") <= 1
    assert np.sum(np.array(["algebra", "algebra", "geometry", "geometry"])[form] == "geometry") >= 1


def test_assemble_test_form_maximizes_information_unconstrained():
    # van der Linden (2005), "Linear Models for Optimal Test Design" (Springer,
    # doi:10.1007/0-387-29054-0): optimal test assembly maximizes the test
    # information function subject to constraints. With no content constraints
    # the optimal fixed-length form is exactly the `length` most informative
    # items, so its total information is the global maximum over every feasible
    # subset. This pins that maximum-information objective so a regression cannot
    # silently degrade assemble_test_form into a feasible-but-suboptimal pick
    # (the existing constrained test only checks feasibility, not optimality).
    from itertools import combinations

    information = np.array([0.10, 0.55, 0.20, 0.80, 0.35, 0.65, 0.05, 0.45])
    length = 3
    form = assemble_test_form(information, length=length)

    top = np.argsort(-information)[:length]
    assert set(form.tolist()) == {int(i) for i in top}
    optimal_total = float(np.sort(information)[-length:].sum())
    assert np.isclose(float(information[form].sum()), optimal_total)
    best_feasible = max(
        float(information[list(subset)].sum())
        for subset in combinations(range(information.size), length)
    )
    assert np.isclose(float(information[form].sum()), best_feasible)


def test_weighted_likelihood_ability_is_finite_for_zero_and_perfect_scores():
    # Warm (1989), "Weighted likelihood estimation of ability in item response
    # theory", Psychometrika 54(3), 427-450 (doi:10.1007/BF02294627): the WLE
    # adds a J(theta)/(2 I(theta)) correction to the score equation so ability
    # stays FINITE for perfect and zero response patterns, where the plain MLE
    # diverges to +/-inf. This pins that zero/perfect-score robustness so a
    # regression cannot silently reduce score_wle to an unbounded MLE.
    a = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 0.8])
    b = np.array([-1.5, -0.8, -0.2, 0.4, 1.0, 1.7])
    responses = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # zero score    -> MLE theta -> -inf
            [1, 1, 1, 0, 0, 0],  # interior pattern
            [1, 1, 1, 1, 1, 1],  # perfect score -> MLE theta -> +inf
        ],
        dtype=float,
    )

    estimate = score_wle(a, b, responses)
    theta = estimate["theta"]

    # Finite, and — the whole point of the Warm correction — strictly interior
    # to the +/-theta_bound clamp rather than saturating it (a raw MLE would peg
    # the bound). The boundary flag is the estimator's own saturation signal.
    assert np.all(np.isfinite(theta))
    assert not np.any(estimate["boundary"])
    assert np.all(np.abs(theta) < 0.9 * 20.0)
    # Monotone in raw score: zero < interior < perfect.
    assert theta[0] < theta[1] < theta[2]
    # Standard errors stay finite and positive at the extremes too.
    assert np.all(np.isfinite(estimate["se"]))
    assert np.all(estimate["se"] > 0.0)


def test_full_information_item_factor_solution_is_stable_and_heywood_free():
    # Bock, R. D., Gibbons, R. D., & Muraki, E. (1988). Full-information item
    # factor analysis. Applied Psychological Measurement, 12(3), 261-280. The
    # multidimensional item-factor model is fit by marginal maximum likelihood,
    # and Bayes constraints on the loadings are needed to keep the solution
    # proper -- i.e. to suppress Heywood cases (divergent/improper estimates).
    # This pins that stability for the MLS2PLM full-information fit: with the
    # penalty (the Bayes constraints) on, two independent same-seed fits return
    # the identical, finite, bounded 2-factor solution rather than drifting or
    # blowing up.
    data = simulate(
        MLS2PLMConfig(n_persons=120, n_dims=2, items_per_dim=4, latent_dim=2, seed=17)
    )
    responses = np.asarray(data.Y, dtype=float)
    config = FitConfig(
        model="MLS2PLM",
        optimizer="adam",
        max_iter=200,
        n_restarts=1,
        latent_dim=2,
        seed=5,
        penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
    )

    first = fit(responses, data.factor_id, config=config)
    second = fit(responses, data.factor_id, config=config)

    # Genuinely the multidimensional full-information model: a 2-factor latent space.
    assert first.params.theta.shape == (120, 2)
    # Stability (1) reproducible: the same seed returns the identical solution.
    assert np.isfinite(first.objective)
    assert np.isclose(first.objective, second.objective)
    assert np.allclose(first.params.b, second.params.b)
    assert np.allclose(first.params.alpha, second.params.alpha)
    assert np.allclose(first.params.theta, second.params.theta)
    # Stability (2) proper / Heywood-free: the Bayes penalty keeps every loading,
    # difficulty, and latent coordinate finite and bounded rather than diverging.
    assert np.all(np.isfinite(first.params.alpha))
    assert np.all(np.isfinite(first.params.b))
    assert np.all(np.isfinite(first.params.theta))
    assert np.max(np.abs(first.params.alpha)) < 20.0
    assert np.max(np.abs(first.params.b)) < 20.0
    assert np.max(np.abs(first.params.theta)) < 20.0
