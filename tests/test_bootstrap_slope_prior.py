"""Actual bootstrap dispatch must preserve the caller's estimator settings."""
from types import SimpleNamespace
import numpy as np
import pytest
import fast_mlsirm.bifactor_bootstrap as bb


def arguments():
    return dict(responses=np.zeros((4, 1)), specific_map=np.array([0]),
                n_cat=2, n_specific=1, n_replicates=1, batch_size=1,
                mc_stopping_ratio=0., compute_budget_seconds=60.,
                q_general=1, q_specific=1, base_seed=0, ci_level=.95,
                max_iter=1, n_starts=1, tol=1e-6, n_jobs=1)


@pytest.mark.parametrize('n_groups', [1, 2])
@pytest.mark.parametrize('mu,sd', [(None, None), (.2, .7)])
def test_real_worker_forwards_prior_and_result_records_it(monkeypatch, n_groups, mu, sd):
    calls = []
    def fit(**kw):
        calls.append(kw)
        shape = (1,) if n_groups == 1 else (2, 1)
        return SimpleNamespace(
            a_general=np.ones(shape), a_specific=np.zeros(shape),
            threshold=np.zeros((*shape, 1)), loglik_trace=np.array([0.]),
            general_mean=np.zeros(n_groups), general_sd=np.ones(n_groups),
            specific_sd=np.ones((n_groups, 1)), converged=True)
    monkeypatch.setattr(bb, 'fit_bifactor_grm', fit)
    monkeypatch.setattr(bb, 'fit_bifactor_grm_multigroup', fit)
    kw = arguments()
    if n_groups == 2:
        kw.update(group_ids=np.array([0, 0, 1, 1]), n_groups=2)
    result = bb.run_bifactor_bootstrap(**kw, slope_prior_mu=mu, slope_prior_sd=sd)
    assert len(calls) == 1
    assert (calls[0]['slope_prior_mu'], calls[0]['slope_prior_sd']) == (mu, sd)
    assert (result.slope_prior_mu, result.slope_prior_sd) == (mu, sd)
    assert result.replicate_ids == result.converged_replicate_ids == (0,)
    assert result.bootstrap_indices_sha256 is None


@pytest.mark.parametrize('mu,sd', [(0., None), (0., -.5), (np.bool_(True), .5)])
def test_bad_prior_rejected_before_worker_dispatch(monkeypatch, mu, sd):
    def worker(*args):
        raise AssertionError('invalid prior reached worker dispatch')
    monkeypatch.setattr(bb, '_fit_single_replicate', worker)
    with pytest.raises(ValueError, match='slope_prior'):
        bb.run_bifactor_bootstrap(**arguments(), slope_prior_mu=mu, slope_prior_sd=sd)


def test_all_failed_run_retains_prior_and_failure_receipt(monkeypatch):
    def fit(**kw):
        assert (kw['slope_prior_mu'], kw['slope_prior_sd']) == (.2, .7)
        raise ValueError('synthetic fit failure')
    monkeypatch.setattr(bb, 'fit_bifactor_grm', fit)
    with pytest.raises(RuntimeError, match='0/1') as raised:
        bb.run_bifactor_bootstrap(**arguments(), slope_prior_mu=.2, slope_prior_sd=.7)
    assert (raised.value.slope_prior_mu, raised.value.slope_prior_sd) == (.2, .7)
    assert raised.value.replicate_ids == (0,)
    assert raised.value.replicate_errors == ('ValueError: synthetic fit failure',)
