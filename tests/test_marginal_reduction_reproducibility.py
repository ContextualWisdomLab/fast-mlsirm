"""Binary64 reproducibility contracts for the marginal-estimator objective."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.estimators.marginal import _item_q, _log_sigmoid


def test_item_q_preserves_elementwise_objective_reduction_order() -> None:
    """Keep the established reduction and prove the rejected split route differs."""
    n_i = np.array([8245.072798436535, 30210347.367437426], dtype=np.float64)
    r_i = np.array([2686.4537974257187, 11191716.404952144], dtype=np.float64)
    eta = np.array([-3.0444188724192074, -31.05286409157143], dtype=np.float64)

    log_positive = _log_sigmoid(eta)
    log_negative = _log_sigmoid(-eta)
    expected = float(
        np.sum(
            r_i * log_positive
            + (n_i - r_i) * log_negative
        )
    )
    split_vdot = float(
        np.vdot(r_i, log_positive)
        + np.vdot(n_i - r_i, log_negative)
    )
    actual = _item_q(
        n_i=n_i,
        r_i=r_i,
        eta=eta,
        alpha_i=1.0,
        b_i=0.0,
        zeta_i=np.empty(0, dtype=np.float64),
        free_alpha=False,
        uses_space=False,
        pen={"lambda_b": 0.0},
    )

    assert split_vdot != expected
    assert split_vdot == np.nextafter(expected, -np.inf)
    assert actual == expected
