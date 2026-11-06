"""Consumer-only regression tests; no native numerical execution.

Basis: fitted-prior provenance preserves the distinction between marginal
likelihood and posterior curvature (Mislevy, 1985, pp. 13–14, eqs. 3.10–3.12).
Reference: Mislevy, R. J. (1985). Bayes modal estimation in item response
models (Research Report RR-85-33). Educational Testing Service.
https://doi.org/10.1002/j.2330-8516.1985.tb00118.x
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fast_mlsirm import bifactor_multigroup, fitstats


def fixture_fit(mu=None, sd=None):
    """Build valid shape-only input, not a simulated numerical fit."""
    return SimpleNamespace(
        a_general=np.ones((1, 2)), a_specific=np.ones((1, 2)),
        threshold=np.zeros((1, 2, 1)), general_mean=np.zeros(1),
        general_sd=np.ones(1), specific_sd=np.ones((1, 1)),
        n_groups=1, n_specific=1, n_cat=2,
        slope_prior_mu=mu, slope_prior_sd=sd,
    )


@pytest.fixture
def recording_core(monkeypatch):
    calls = []

    def record(*args):
        calls.append(args)
        return dict(labels=['a_general:0'], information=[1.0], vcov=None,
                    se=None, positive_definite=False, non_pd_reason='fixture')

    monkeypatch.setattr(fitstats, '_core_module', lambda: SimpleNamespace(
        bifactor_multigroup_oakes_se=record))
    return calls


def oakes(fit):
    return bifactor_multigroup.bifactor_multigroup_oakes_se(
        fit, np.array([[0, 1], [1, 0]]), np.zeros(2, dtype=int),
        np.zeros(2, dtype=int), np.ones(2, dtype=bool),
        q_general=7, q_specific=7, fd_step=1e-5,
    )


@pytest.mark.parametrize('prior', [(None, None), (np.float64(0.1), np.float64(0.5))])
def test_joint_oakes_preserves_fitted_prior_on_result(recording_core, prior):
    result = oakes(fixture_fit(*prior))
    assert recording_core[0][-2:] == prior
    assert (result.slope_prior_mu, result.slope_prior_sd) == prior
    for value in (result.slope_prior_mu, result.slope_prior_sd):
        assert value is None or type(value) is float


@pytest.mark.parametrize('mu,sd,message', [
    (0.1, None, 'provided together'),
    (None, 0.5, 'provided together'),
    (float('nan'), 0.5, 'slope_prior_mu'),
    (0.1, 0.0, 'slope_prior_sd'),
    (0.1, float('inf'), 'slope_prior_sd'),
    (np.bool_(True), 0.5, 'slope_prior_mu'),
    (0.1, np.bool_(True), 'slope_prior_sd'),
    (0.1+0j, 0.5, 'slope_prior_mu'),
    (np.array(0.1), 0.5, 'slope_prior_mu'),
])
def test_joint_oakes_invalid_fitted_prior_rejected_before_dispatch(recording_core, mu, sd, message):
    with pytest.raises(ValueError, match=message):
        oakes(fixture_fit(mu, sd))
    assert recording_core == []
