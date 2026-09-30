import numpy as np
import pytest
import time

def test_marginal_sum_e_v2_parity():
    np.random.seed(20261024)
    cluster_post = np.random.randn(500, 121)
    u_nodes = np.random.randn(121)

    t0 = time.perf_counter()
    for _ in range(100):
        old_res = float((cluster_post * u_nodes[None, :] ** 2).sum())
    t1 = time.perf_counter()

    for _ in range(100):
        new_res = float(np.vdot(cluster_post.sum(axis=0), u_nodes ** 2))
    t2 = time.perf_counter()

    assert np.isclose(old_res, new_res)
    assert t2 - t1 < t1 - t0

def test_marginal_grad_d_parity():
    np.random.seed(20261024)
    resid = np.random.randn(20, 100, 21, 5)
    w_cov = np.random.randn(20, 100)
    w_bcast = w_cov[:, :, None, None]

    t0 = time.perf_counter()
    for _ in range(100):
        old_res = float((resid * w_bcast).sum())
    t1 = time.perf_counter()

    for _ in range(100):
        new_res = float(np.vdot(resid.sum(axis=(2, 3)), w_cov))
    t2 = time.perf_counter()

    assert np.isclose(old_res, new_res)
    assert t2 - t1 < t1 - t0

def test_marginal_info_d_parity():
    np.random.seed(20261024)
    n_all = np.random.rand(20, 100, 21, 5)
    prob = np.random.rand(20, 100, 21, 5)
    w_cov = np.random.randn(20, 100)
    w_bcast = w_cov[:, :, None, None]

    t0 = time.perf_counter()
    for _ in range(100):
        old_res = float((n_all * prob * (1.0 - prob) * w_bcast * w_bcast).sum())
    t1 = time.perf_counter()

    for _ in range(100):
        new_res = float(np.vdot((n_all * prob * (1.0 - prob)).sum(axis=(2, 3)), w_cov * w_cov))
    t2 = time.perf_counter()

    assert np.isclose(old_res, new_res)
    assert t2 - t1 < t1 - t0

def test_marginal_m_parity():
    np.random.seed(20261024)
    w = np.random.randn(121, 5)
    theta_g = np.random.randn(121)

    t0 = time.perf_counter()
    for _ in range(100):
        old_m1 = float((w * theta_g[:, None]).sum())
        old_m2 = float((w * (theta_g**2)[:, None]).sum())
    t1 = time.perf_counter()

    for _ in range(100):
        w_sum_ax = w.sum(axis=1)
        new_m1 = float(np.vdot(w_sum_ax, theta_g))
        new_m2 = float(np.vdot(w_sum_ax, theta_g**2))
    t2 = time.perf_counter()

    assert np.isclose(old_m1, new_m1)
    assert np.isclose(old_m2, new_m2)
    assert t2 - t1 < t1 - t0
