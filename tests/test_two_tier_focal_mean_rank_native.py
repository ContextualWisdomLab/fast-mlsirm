"""Native free-mean nullspace rejection; Cai (2010), p.589 eq.11.

A*v=0 implies identical predictors under mu and mu+v. Rank guard mechanism:
LAPACK DGETF2 Purpose/INFO, https://www.netlib.org/lapack/double/dgetf2.f.
Its normalized dimension*EPSILON guard is an implementation choice, not
DGELSY effective rank or proof of variance/joint statistical identification.
"""

import numpy as np
import pytest

from fast_mlsirm import fit_two_tier_grm_focal_orthogonal


def test_native_focal_fit_rejects_nonidentified_free_means():
    """Fail on old compiled code that accepts two identical loading columns."""
    with pytest.raises(ValueError, match="column rank"):
        fit_two_tier_grm_focal_orthogonal(
            responses=np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 0, 0]]),
            primary_map=np.ones((4, 1), dtype=bool),
            specific_map=np.zeros(4, dtype=np.int64),
            a_primary=np.array([[0.8], [1.2], [1.0], [0.6]]),
            a_specific=np.array([0.8, 1.2, 1.0, 0.6]),
            threshold=np.tile([0.8, -0.6], (4, 1)),
            latent_mean=np.zeros(2),
            latent_sd=np.ones(2),
            n_cat=3,
            n_primary=1,
            n_specific=1,
            q_primary=5,
            q_specific=5,
            max_iter=1,
            tol=1e6,
        )
