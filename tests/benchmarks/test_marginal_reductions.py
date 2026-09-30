import numpy as np
import pytest

def test_marginal_sum_e_v2_parity():
    cluster_post = np.random.randn(500, 121)
    u_nodes = np.random.randn(121)
    old_res = float((cluster_post * u_nodes[None, :] ** 2).sum())
    new_res = float(np.vdot(cluster_post.sum(axis=0), u_nodes ** 2))
    assert np.isclose(old_res, new_res)

def test_marginal_grad_d_parity():
    resid = np.random.randn(20, 100, 21, 5)
    w_cov = np.random.randn(20, 100)
    w_bcast = w_cov[:, :, None, None]
    old_res = float((resid * w_bcast).sum())
    new_res = float(np.vdot(resid.sum(axis=(2, 3)), w_cov))
    assert np.isclose(old_res, new_res)

def test_marginal_info_d_parity():
    n_all = np.random.rand(20, 100, 21, 5)
    prob = np.random.rand(20, 100, 21, 5)
    w_cov = np.random.randn(20, 100)
    w_bcast = w_cov[:, :, None, None]
    old_res = float((n_all * prob * (1.0 - prob) * w_bcast * w_bcast).sum())
    new_res = float(np.vdot((n_all * prob * (1.0 - prob)).sum(axis=(2, 3)), w_cov * w_cov))
    assert np.isclose(old_res, new_res)

def test_marginal_m_parity():
    w = np.random.randn(121, 5)
    theta_g = np.random.randn(121)
    old_m1 = float((w * theta_g[:, None]).sum())
    old_m2 = float((w * (theta_g**2)[:, None]).sum())

    w_sum_ax = w.sum(axis=1)
    new_m1 = float(np.vdot(w_sum_ax, theta_g))
    new_m2 = float(np.vdot(w_sum_ax, theta_g**2))

    assert np.isclose(old_m1, new_m1)
    assert np.isclose(old_m2, new_m2)
