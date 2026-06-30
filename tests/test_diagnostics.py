import pytest
import numpy as np
from fast_mlsirm.diagnostics import recovery_report, align_latent_space, predict_proba
from fast_mlsirm.types import MLSIRMParams

def test_align_latent_space_no_space():
    truth_xi = np.zeros((0, 0))
    truth_zeta = np.zeros((0, 0))
    est_xi = np.zeros((0, 0))
    est_zeta = np.zeros((0, 0))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        aligned_xi, aligned_zeta = align_latent_space(truth_xi, truth_zeta, est_xi, est_zeta)
    assert aligned_xi.size == 0

def test_predict_proba():
    p = MLSIRMParams(
        theta=np.array([[0.0], [0.0]]),
        alpha=np.array([0.0]),
        b=np.array([0.0]),
        xi=np.array([[0.0], [0.0]]),
        zeta=np.array([[0.0]]),
        tau=0.0,
    )
    factor_id = np.array([0])
    prob = predict_proba(p, factor_id, model="MLS2PLM")
    assert prob.shape == (2, 1)

def test_recovery_report_no_space():
    truth = MLSIRMParams(theta=np.zeros((2, 1)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((0, 0)), zeta=np.zeros((0, 0)), tau=0.0)
    est = MLSIRMParams(theta=np.zeros((2, 1)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((0, 0)), zeta=np.zeros((0, 0)), tau=0.0)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        report = recovery_report(truth, est)
        assert np.isnan(report.summary["distance_rmse"])

def test_recovery_report_missing_alpha():
    truth = MLSIRMParams(theta=np.zeros((2, 1)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 1)), zeta=np.zeros((2, 1)), tau=0.0)
    est = MLSIRMParams(theta=np.zeros((2, 1)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 1)), zeta=np.zeros((2, 1)), tau=0.0)
    report = recovery_report(truth, est)
    assert report.metrics["a_rmse"] == 0.0

def test_align_latent_space_invalid_method():
    with pytest.raises(ValueError):
        align_latent_space(np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)), method="invalid")
def test_subset_params():
    from fast_mlsirm.diagnostics import _subset_params
    p = MLSIRMParams(
        theta=np.array([[1.0], [2.0]]),
        alpha=np.array([1.0, 2.0]),
        b=np.array([1.0, 2.0]),
        xi=np.array([[1.0], [2.0]]),
        zeta=np.array([[1.0], [2.0]]),
        tau=0.0,
    )
    sub = _subset_params(p, persons=np.array([1]), items=np.array([1]))
    assert sub.theta.shape == (1, 1)
    assert sub.theta[0,0] == 2.0
def test_predict_proba_subset():
    from fast_mlsirm.diagnostics import predict_proba
    p = MLSIRMParams(
        theta=np.array([[0.0], [0.0], [0.0]]),
        alpha=np.array([0.0, 0.0, 0.0]),
        b=np.array([0.0, 0.0, 0.0]),
        xi=np.array([[0.0], [0.0], [0.0]]),
        zeta=np.array([[0.0], [0.0], [0.0]]),
        tau=0.0,
    )
    factor_id = np.array([0, 0, 0])
    prob = predict_proba(p, factor_id, persons=np.array([0, 1]), items=np.array([1, 2]))
    assert prob.shape == (2, 2)
