"""G-only expected raw API controls; no numerical kernels in Python tests."""
from types import SimpleNamespace
import numpy as np
import pytest


def bank():
    return SimpleNamespace(n_primary=2, n_specific=1, n_cat=3,
        a_primary=np.array([[.7, 1.2], [1.1, -.6]]),
        a_specific=np.array([.4, 0.]), threshold=np.array([[.8, -.5], [.4, -.9]]),
        phi=np.eye(2), theta_p_eap=np.array([[-.8, -9.], [0., 0.], [.9, 8.]]))


def test_new_api_available():
    from fast_mlsirm.two_tier_g_only import expected_raw_g_only
    assert callable(expected_raw_g_only)


def test_unknown_keywords_rejected():
    from fast_mlsirm.two_tier_g_only import expected_raw_g_only
    with pytest.raises(TypeError):
        expected_raw_g_only(bank(), [0, -1], mean_w=0., sd_w=1., specific_sd=[1.],
            q_wording=121, q_specific=121, device="gpu", posterior_average=True)


@pytest.mark.parametrize("bad", [0, True, 1.5, "121"])
def test_node_types_rejected(bad):
    from fast_mlsirm.two_tier_g_only import expected_raw_g_only
    with pytest.raises(ValueError):
        expected_raw_g_only(bank(), [0, -1], mean_w=0., sd_w=1., specific_sd=[1.],
            q_wording=bad, q_specific=121, device="gpu")


@pytest.mark.parametrize("change", ["phi", "n_primary", "shape", "map"])
def test_invalid_bank_refused_before_native(monkeypatch, change):
    from fast_mlsirm.two_tier_g_only import expected_raw_g_only
    from fast_mlsirm import _core
    def deny(*args, **kwargs):
        pytest.fail("native called before validation")
    monkeypatch.setattr(_core, "two_tier_expected_raw_g_only", deny)
    fit=bank(); smap=[0, -1]
    if change == "phi": fit.phi[0, 1] = .2
    if change == "n_primary": fit.n_primary = 1
    if change == "shape": fit.theta_p_eap = np.zeros((3, 1))
    if change == "map": smap = [0, -2]
    with pytest.raises(ValueError):
        expected_raw_g_only(fit, smap, mean_w=0., sd_w=1., specific_sd=[1.],
            q_wording=121, q_specific=121, device="gpu")


def test_native_receipt_controls_not_silently_changed(monkeypatch):
    from fast_mlsirm.two_tier_g_only import expected_raw_g_only
    from fast_mlsirm import _core
    def reply(*args):
        return dict(scores=np.array([1.,1.,1.]),dispatches=1,cpu_numerical_fallbacks=0,
                    estimand="E[T|G=g]",q_wording=15,q_specific=15,backend="Metal",device="fake")
    monkeypatch.setattr(_core,"two_tier_expected_raw_g_only",reply)
    with pytest.raises(RuntimeError,match="receipt"):
        expected_raw_g_only(bank(),[0,-1],mean_w=0.,sd_w=1.,specific_sd=[1.],
                            q_wording=121,q_specific=241,device="gpu")
