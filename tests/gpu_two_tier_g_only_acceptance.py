"""Physical GPU acceptance, selected explicitly under shared supervisor lease.

Run pytest tests/gpu_two_tier_g_only_acceptance.py. No CPU fallback or skip.
"""
import os
from types import SimpleNamespace
import numpy as np
import pytest
from fast_mlsirm import _core
from fast_mlsirm.two_tier_g_only import expected_raw_g_only


def call(**changes):
    assert os.environ.get("FMLS_E_GPU_LEASED") == "1"
    args = dict(a_primary=np.zeros(48), a_specific=np.zeros(24),
                threshold=np.zeros(24), general=np.array([-1., 0., 1.]),
                specific_map=np.full(24, -1, dtype=np.int64), n_cat=2,
                n_specific=0, mean_w=0., sd_w=1., specific_sd=np.array([]),
                primary_covariance=0., q_wording=15, q_specific=15, device="gpu")
    args.update(changes)
    return _core.two_tier_expected_raw_g_only(**args)


def test_binding_exists():
    assert callable(getattr(_core, "two_tier_expected_raw_g_only", None))


@pytest.mark.parametrize("changes", [dict(device="cpu"), dict(device="auto"),
    dict(primary_covariance=.1), dict(q_wording=0),
    dict(specific_map=np.full(24, 2**40, dtype=np.int64)),
    dict(general=np.array([np.nan]))])
def test_boundary_refusals(changes):
    with pytest.raises(ValueError):
        call(**changes)


@pytest.mark.parametrize("nodes", [15, 121, 241])
def test_constant_bank_native(nodes):
    out = call(q_wording=nodes, q_specific=nodes)
    np.testing.assert_array_equal(out["scores"], [12., 12., 12.])
    assert out["dispatches"] == 1
    assert out["cpu_numerical_fallbacks"] == 0
    assert out["q_wording"] == out["q_specific"] == nodes
    assert out["estimand"] == "E[T|G=g]"


def fit():
    return SimpleNamespace(n_primary=2,n_specific=1,n_cat=3,phi=np.eye(2),
        a_primary=np.array([[.7,1.2],[1.1,-.6]]),a_specific=np.array([.4,0.]),
        threshold=np.array([[.8,-.5],[.4,-.9]]),
        theta_p_eap=np.array([[-.8,-9.],[0.,0.],[.9,8.]]))


def score(f,**kwargs):
    assert os.environ.get("FMLS_E_GPU_LEASED")=="1"
    settings=dict(mean_w=.3,sd_w=.8,specific_sd=[.6],q_wording=121,q_specific=241,device="gpu")
    settings.update(kwargs)
    return expected_raw_g_only(f,[0,-1],**settings)["scores"]


def test_w_eap_unused_but_nuisance_settings_have_effect():
    f=fit(); first=score(f)
    f.theta_p_eap[:,1]=[100.,-200.,30.]
    np.testing.assert_array_equal(score(f),first)
    assert np.max(np.abs(score(f,mean_w=1.3)-first)) > .01
    assert np.max(np.abs(score(f,sd_w=2.)-first)) > .001
    assert np.max(np.abs(score(f,specific_sd=[2.])-first)) > .001
    f.theta_p_eap=f.theta_p_eap[[2,0,1]]
    np.testing.assert_array_equal(score(f),first[[2,0,1]])


def test_extreme_finite_products_rejected_by_binding():
    with pytest.raises(ValueError,match="unsafe f32 predictor range"):
        call(a_primary=np.array([3e38,-3e38]),a_specific=np.array([0.]),
            threshold=np.array([0.]),general=np.array([2.]),
            specific_map=np.array([-1],dtype=np.int64),mean_w=2.,q_wording=1,q_specific=1)
