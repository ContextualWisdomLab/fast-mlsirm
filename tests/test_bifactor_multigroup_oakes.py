"""Public joint Oakes entry point: exact single-group delegation and validation.

Basis: Oakes (1999, eq. 6, p. 480); MAP curvature per Mislevy (1986).
References (APA 7th): Oakes, D. (1999). Direct calculation of the
information matrix via the EM algorithm. *Journal of the Royal Statistical
Society: Series B, 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188
Mislevy, R. J. (1986). Bayes modal estimation in item response models.
*Psychometrika, 51*(2), 177–195. https://doi.org/10.1007/BF02293979
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fast_mlsirm import bifactor_multigroup_oakes_se, bifactor_oakes_se


def test_single_group_exact_and_validation():
    y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    smap = np.array([0, 0])
    ag = np.array([0.8, 1.0])
    as_ = np.array([0.5, 0.7])
    th = np.array([[0.1], [-0.2]])
    fit = SimpleNamespace(
        a_general=ag[None, :], a_specific=as_[None, :],
        threshold=th[None, :, :], general_mean=np.array([0.0]),
        general_sd=np.array([1.0]), specific_sd=np.array([[1.0]]),
        n_groups=1, n_specific=1, n_cat=2,
        slope_prior_mu=None, slope_prior_sd=None,
    )
    kwargs = dict(q_general=3, q_specific=3, fd_step=1e-5)
    single = bifactor_oakes_se(ag, as_, th, y, smap, 2, 1, **kwargs)
    joint = bifactor_multigroup_oakes_se(
        fit, y, np.zeros(4, dtype=int), smap, np.ones(2, dtype=bool), **kwargs)
    assert joint.labels == single.labels
    assert np.array_equal(joint.information, single.information)
    assert joint.positive_definite == single.positive_definite
    if joint.vcov is None:
        assert single.vcov is None and single.se is None and joint.se is None
    else:
        assert np.array_equal(joint.vcov, single.vcov)
        assert np.array_equal(joint.se, single.se)
    with pytest.raises(ValueError, match="quadrature"):
        bifactor_multigroup_oakes_se(
            fit, y, np.zeros(4, dtype=int), smap, None,
            q_general=0, q_specific=3, fd_step=1e-5)
    with pytest.raises(ValueError, match="group"):
        bifactor_multigroup_oakes_se(
            fit, y, np.ones(4, dtype=int), smap, None, **kwargs)


def test_map_fit_adds_slope_prior_curvature_per_estimated_slope():
    y = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [1, 1], [0, 1]])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    smap = np.array([0, 0])
    ag = np.array([[0.8, 1.0], [0.8, 1.1]])
    as_ = np.array([[0.5, 0.7], [0.5, 0.8]])
    ml_fit = SimpleNamespace(
        a_general=ag, a_specific=as_,
        threshold=np.array([[[0.1], [-0.2]], [[0.1], [-0.1]]]),
        general_mean=np.array([0.0, 0.2]), general_sd=np.array([1.0, 1.1]),
        specific_sd=np.array([[1.0], [1.0]]),
        n_groups=2, n_specific=1, n_cat=2,
        slope_prior_mu=None, slope_prior_sd=None,
    )
    mu, sd = 0.1, 0.5
    map_fit = SimpleNamespace(**{**vars(ml_fit), "slope_prior_mu": mu, "slope_prior_sd": sd})
    kwargs = dict(q_general=5, q_specific=5, fd_step=1e-6)
    anchor = np.array([True, False])
    ml = bifactor_multigroup_oakes_se(ml_fit, y, group, smap, anchor, **kwargs)
    mp = bifactor_multigroup_oakes_se(map_fit, y, group, smap, anchor, **kwargs)
    assert mp.labels == ml.labels

    def curvature(a):
        z = (np.log(abs(a)) - mu) / sd
        return (1.0 / sd**2 - 1.0 - z / sd) / a**2

    slopes = {
        "a_general:0": ag[0, 0], "a_specific:0": as_[0, 0],
        "a_general:0:1": ag[0, 1], "a_specific:0:1": as_[0, 1],
        "a_general:1:1": ag[1, 1], "a_specific:1:1": as_[1, 1],
    }
    want = np.diag([curvature(slopes[lab]) if lab in slopes else 0.0 for lab in ml.labels])
    np.testing.assert_allclose(mp.information - ml.information, want, rtol=1e-10, atol=1e-10)
    bad = SimpleNamespace(**{**vars(map_fit), "slope_prior_sd": None})
    with pytest.raises(ValueError, match="together"):
        bifactor_multigroup_oakes_se(bad, y, group, smap, anchor, **kwargs)
