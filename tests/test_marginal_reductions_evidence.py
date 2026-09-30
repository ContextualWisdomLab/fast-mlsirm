"""Evidence for parity of optimized reductions in the marginal estimator."""

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit
from fast_mlsirm.reference import fit_reference
from tests.test_marginal_parity import _assert_close, _simulate

pytestmark = pytest.mark.skipif(
    pytest.importorskip("fast_mlsirm._core", reason="compiled core required") is None,
    reason="compiled core required",
)

def test_marginal_reductions_parity_evidence_multilevel() -> None:
    """Ensure optimization of multilevel reduction maintains Rust parity."""
    y, fid = _simulate(seed=1024, n_persons=100, n_items=12, n_dims=2, latent_dim=2)
    cluster_id = np.arange(len(y)) % 10
    cfg_args = dict(model="MLS2PLM", estimator="mmle", max_iter=3, rust_device="cpu", q_theta=123, q_xi=7, q_u=125)

    r = fit(y, fid, FitConfig(backend="rust", **cfg_args), cluster_id=cluster_id)
    n = fit_reference(y, fid, FitConfig(backend="numpy", **cfg_args), cluster_id=cluster_id)

    _assert_close(r, n, tol=1e-8)
    np.testing.assert_allclose(r.population["sigma_u"], n.population["sigma_u"], atol=1e-8)

def test_marginal_reductions_parity_evidence_covariate() -> None:
    """Ensure optimization of covariate reduction maintains Rust parity."""
    y, fid = _simulate(seed=2048, n_persons=100, n_items=12, n_dims=2, latent_dim=2)
    group_id = np.arange(len(y)) % 3
    w_cov = np.random.RandomState(42).randn(3, y.shape[1])
    cfg_args = dict(model="MLS2PLM", estimator="mmle", max_iter=3, rust_device="cpu", q_theta=123, q_xi=7, q_u=125)

    r = fit(y, fid, FitConfig(backend="rust", **cfg_args), group_id=group_id, covariate={"w": w_cov})
    n = fit_reference(y, fid, FitConfig(backend="numpy", **cfg_args), group_id=group_id, covariate={"w": w_cov})

    _assert_close(r, n, tol=1e-8)
    np.testing.assert_allclose(r.population["mu"], n.population["mu"], atol=1e-8)
    np.testing.assert_allclose(r.population["sigma"], n.population["sigma"], atol=1e-8)
