"""Public joint Oakes entry point: exact single-group delegation and validation.

Basis: Oakes (1999, eq. 6, p. 480). Reference (APA 7th): Oakes, D.
(1999). Direct calculation of the information matrix via the EM algorithm.
*Journal of the Royal Statistical Society: Series B, 61*(2), 479–482.
https://doi.org/10.1111/1467-9868.00188
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
