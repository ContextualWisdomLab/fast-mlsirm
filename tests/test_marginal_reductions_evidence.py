"""Evidence for parity of optimized reductions in the marginal estimator."""

import math

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.estimators import marginal
from fast_mlsirm.fit import fit
from fast_mlsirm.reference import fit_reference
from tests.test_marginal_parity import _assert_close, _simulate

pytestmark = pytest.mark.skipif(
    pytest.importorskip("fast_mlsirm._core", reason="compiled core required") is None,
    reason="compiled core required",
)


def test_reduction_helpers_preserve_hand_derived_values() -> None:
    """Catch any changed axis or weight in the three optimized reductions."""
    cluster_post = np.array([[0.25, 0.75], [0.5, 0.5]])
    u_nodes = np.array([-1.0, 2.0])
    assert marginal._multilevel_second_moment(cluster_post, u_nodes) == 5.75

    residual = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    expected_count = np.ones_like(residual)
    probability = np.full_like(residual, 0.5)
    covariate = np.array([[2.0]])
    score, information = marginal._covariate_score_information(
        residual,
        expected_count,
        probability,
        covariate,
    )
    assert score == 20.0
    assert information == 4.0

    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    nodes = np.array([10.0, 20.0])
    assert marginal._weighted_population_moments(weights, nodes) == (
        10.0,
        170.0,
        3100.0,
    )


def test_covariate_reduction_respects_float64_forward_error_bound() -> None:
    """Bound the changed reduction order by the standard summation error model."""
    random_generator = np.random.default_rng(2310)
    shape = (3, 5, 7, 11)
    residual = random_generator.normal(size=shape)
    expected_count = random_generator.uniform(0.5, 4.0, size=shape)
    probability = random_generator.uniform(0.1, 0.9, size=shape)
    covariate = random_generator.normal(size=shape[:2])

    score, information = marginal._covariate_score_information(
        residual,
        expected_count,
        probability,
        covariate,
    )
    score_terms = residual * covariate[:, :, None, None]
    information_terms = (
        expected_count
        * probability
        * (1.0 - probability)
        * covariate[:, :, None, None] ** 2
    )

    for observed, terms in (
        (score, score_terms),
        (information, information_terms),
    ):
        oracle = math.fsum(float(value) for value in terms.flat)
        unit_roundoff = np.finfo(np.float64).eps / 2.0
        gamma = terms.size * unit_roundoff / (
            1.0 - terms.size * unit_roundoff
        )
        forward_error_bound = gamma * math.fsum(
            abs(float(value)) for value in terms.flat
        )
        assert abs(observed - oracle) <= forward_error_bound


def test_changed_reductions_recover_their_generating_sufficient_statistics() -> None:
    """Recover standard-normal moments and the covariate score root.

    A 121-node Gauss-Hermite rule integrates these degree-two normal moments
    exactly.  At the generating unit covariate coefficient, the expected
    Bernoulli residual is zero and Fisher information is strictly positive, so
    the score equation has its unique root at the truth.  Comparisons use the
    standard floating-point summation-error model rather than empirical
    tolerances.
    """
    nodes, weights = marginal._gh(121)
    unit_roundoff = np.finfo(np.float64).eps / 2.0

    multilevel_second_moment = marginal._multilevel_second_moment(
        weights[None, :],
        nodes,
    )
    population_weight, population_mean, population_second_moment = (
        marginal._weighted_population_moments(weights[:, None], nodes)
    )

    covariate = np.array([[-1.0, 1.0], [1.0, -1.0]])
    expected_count = np.broadcast_to(
        weights[None, None, :, None],
        (2, 2, weights.size, 1),
    )
    linear_predictor_at_truth = (
        nodes[None, None, :, None] + covariate[:, :, None, None]
    )
    probability = 1.0 / (1.0 + np.exp(-linear_predictor_at_truth))
    residual_at_truth = expected_count * probability - expected_count * probability
    score_at_truth, information_at_truth = marginal._covariate_score_information(
        residual_at_truth,
        expected_count,
        probability,
        covariate,
    )

    supplied_rule_moments = (
        math.fsum(float(value) for value in weights),
        math.fsum(float(value) for value in weights * nodes),
        math.fsum(float(value) for value in weights * nodes**2),
    )
    assert supplied_rule_moments == (1.0, 0.0, 1.0)

    observed_and_terms = (
        (multilevel_second_moment, weights * nodes**2),
        (population_weight, weights),
        (population_mean, weights * nodes),
        (population_second_moment, weights * nodes**2),
    )
    fsum_oracles = (1.0, 1.0, 0.0, 1.0)
    for (observed, terms), oracle in zip(observed_and_terms, fsum_oracles):
        gamma = terms.size * unit_roundoff / (1.0 - terms.size * unit_roundoff)
        bound = gamma * math.fsum(abs(float(value)) for value in terms)
        assert abs(observed - oracle) <= bound

    information_terms = (
        expected_count
        * probability
        * (1.0 - probability)
        * covariate[:, :, None, None] ** 2
    )
    information_oracle = math.fsum(float(value) for value in information_terms.flat)
    information_gamma = information_terms.size * unit_roundoff / (
        1.0 - information_terms.size * unit_roundoff
    )
    information_bound = information_gamma * math.fsum(
        abs(float(value)) for value in information_terms.flat
    )
    assert score_at_truth == 0.0
    assert information_at_truth > 0.0
    assert abs(information_at_truth - information_oracle) <= information_bound


def test_marginal_reductions_parity_evidence_multilevel() -> None:
    """Ensure optimization of multilevel reduction maintains Rust parity."""
    y, fid = _simulate(seed=1024, n_persons=100, n_items=12, n_dims=2, latent_dim=2)
    cluster_id = np.arange(len(y)) % 10
    cfg_args = dict(model="MLS2PLM", estimator="mmle", max_iter=3, rust_device="cpu", q_theta=123, q_xi=7, q_u=125)

    r = fit(y, fid, FitConfig(backend="rust", **cfg_args), cluster_id=cluster_id)
    n = fit_reference(y, fid, FitConfig(backend="numpy", **cfg_args), cluster_id=cluster_id)

    _assert_close(r, n, tol=1e-8)
    np.testing.assert_allclose(r.population["sigma_u"], n.population["sigma_u"], atol=1e-8)

def test_marginal_reductions_parity_evidence_covariate() -> None:
    """Ensure optimization of covariate reduction maintains Rust parity."""
    y, fid = _simulate(seed=2048, n_persons=100, n_items=12, n_dims=2, latent_dim=2)
    group_id = np.arange(len(y)) % 3
    w_cov = np.random.RandomState(42).randn(3, y.shape[1])
    cfg_args = dict(model="MLS2PLM", estimator="mmle", max_iter=3, rust_device="cpu", q_theta=123, q_xi=7, q_u=125)

    r = fit(y, fid, FitConfig(backend="rust", **cfg_args), group_id=group_id, covariate={"w": w_cov})
    n = fit_reference(y, fid, FitConfig(backend="numpy", **cfg_args), group_id=group_id, covariate={"w": w_cov})

    _assert_close(r, n, tol=1e-8)
    np.testing.assert_allclose(r.population["mu"], n.population["mu"], atol=1e-8)
    np.testing.assert_allclose(r.population["sigma"], n.population["sigma"], atol=1e-8)
