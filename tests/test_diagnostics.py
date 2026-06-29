import numpy as np
import pytest

from fast_mlsirm.diagnostics import align_latent_space

def test_align_latent_space_invalid_method():
    true_xi = np.random.randn(10, 2)
    true_zeta = np.random.randn(5, 2)
    est_xi = np.random.randn(10, 2)
    est_zeta = np.random.randn(5, 2)

    with pytest.raises(ValueError, match="only procrustes alignment is supported"):
        align_latent_space(true_xi, true_zeta, est_xi, est_zeta, method="invalid")
