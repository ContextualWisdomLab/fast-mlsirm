import numpy as np
from fast_mlsirm.math import standardize

def test_standardize_zero_variance():
    res = standardize(np.array([1.0, 1.0]))
    assert np.array_equal(res, np.zeros(2))

def test_normalize_latent_positions_empty():
    from fast_mlsirm.types import MLSIRMParams
    from fast_mlsirm.math import normalize_latent_positions
    p = MLSIRMParams(
        theta=np.zeros((2, 2)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((0, 0)),
        zeta=np.zeros((0, 0)),
        tau=0.0,
    )
    res = normalize_latent_positions(p)
    assert res.xi.size == 0
