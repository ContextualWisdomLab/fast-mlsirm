"""Numerical parity checks for allocation-reducing marginal estimators."""

import numpy as np


def test_marginal_sum_e_v2_parity() -> None:
    """Axis-first reduction preserves the scalar second moment."""
    rng = np.random.default_rng(20261024)
    cluster_post = rng.standard_normal((500, 121))
    u_nodes = rng.standard_normal(121)

    old_result = float((cluster_post * u_nodes[None, :] ** 2).sum())
    new_result = float(np.vdot(cluster_post.sum(axis=0), u_nodes**2))

    np.testing.assert_allclose(new_result, old_result)


def test_marginal_grad_d_parity() -> None:
    """Axis-first reduction preserves the delta gradient."""
    rng = np.random.default_rng(20261024)
    residual = rng.standard_normal((20, 100, 21, 5))
    covariate_weights = rng.standard_normal((20, 100))
    broadcast_weights = covariate_weights[:, :, None, None]

    old_result = float((residual * broadcast_weights).sum())
    new_result = float(np.vdot(residual.sum(axis=(2, 3)), covariate_weights))

    np.testing.assert_allclose(new_result, old_result)


def test_marginal_info_d_parity() -> None:
    """Axis-first reduction preserves the delta information."""
    rng = np.random.default_rng(20261024)
    counts = rng.random((20, 100, 21, 5))
    probabilities = rng.random((20, 100, 21, 5))
    covariate_weights = rng.standard_normal((20, 100))
    broadcast_weights = covariate_weights[:, :, None, None]

    old_result = float(
        (
            counts
            * probabilities
            * (1.0 - probabilities)
            * broadcast_weights
            * broadcast_weights
        ).sum()
    )
    new_result = float(
        np.vdot(
            (counts * probabilities * (1.0 - probabilities)).sum(axis=(2, 3)),
            covariate_weights * covariate_weights,
        )
    )

    np.testing.assert_allclose(new_result, old_result)


def test_marginal_weighted_moment_parity() -> None:
    """Axis-first reduction preserves the first two weighted moments."""
    rng = np.random.default_rng(20261024)
    weights = rng.standard_normal((121, 5))
    theta_nodes = rng.standard_normal(121)

    old_first = float((weights * theta_nodes[:, None]).sum())
    old_second = float((weights * (theta_nodes**2)[:, None]).sum())

    summed_weights = weights.sum(axis=1)
    new_first = float(np.vdot(summed_weights, theta_nodes))
    new_second = float(np.vdot(summed_weights, theta_nodes**2))

    np.testing.assert_allclose(new_first, old_first)
    np.testing.assert_allclose(new_second, old_second)
