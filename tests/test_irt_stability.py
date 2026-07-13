import numpy as np

from fast_mlsirm import FitConfig, MLS2PLMConfig, MLSIRMParams, PenaltyConfig, recovery_report, simulate
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
        # Finite differences with step=1e-4 amplify f32 noise from the wgpu GPU
        # path into a failed PSD check; the Hessian must come from the f64 CPU path.
        rust_device="cpu",
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
