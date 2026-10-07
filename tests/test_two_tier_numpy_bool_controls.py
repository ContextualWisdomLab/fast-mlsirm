"""Boolean scalar admission only; recording results are not numerical fits."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from test_two_tier_expected_raw_layout import seam

@pytest.mark.parametrize('helper',['_finite_integer_control','_positive_real_control','_u64_seed'])
@pytest.mark.parametrize('value',[True,False,np.bool_(True),np.bool_(False)],ids=['bool-true','bool-false','numpy-true','numpy-false'])
def test_boolean_scalar_controls_are_rejected(seam,helper,value):
    module,calls=seam
    with pytest.raises(ValueError):
        getattr(module,helper)(value) if helper=='_u64_seed' else getattr(module,helper)(value,'control')
    assert calls==[]

@pytest.mark.parametrize('helper,value,expected',[
    ('_finite_integer_control',121,121),('_finite_integer_control',np.int64(241),241),
    ('_positive_real_control',1e-6,1e-6),('_positive_real_control',np.float64(1e-4),1e-4),
    ('_u64_seed',17,17),('_u64_seed',np.uint64(17),17)])
def test_nonboolean_controls_keep_existing_values(seam,helper,value,expected):
    module,calls=seam
    got=getattr(module,helper)(value) if helper=='_u64_seed' else getattr(module,helper)(value,'control')
    assert got==expected and calls==[]

@pytest.mark.parametrize('route',['score-q','fit-seed','oakes-step'])
def test_numpy_boolean_rejected_before_public_dispatch(seam,monkeypatch,route):
    module,unused=seam;calls=[]
    ag=np.array([[1.2],[0.8]]);asp=np.array([0.5,0.4]);th=np.array([[0.2],[-0.2]])
    y=np.array([[0,1],[1,0]]);pm=np.ones((2,1),dtype=bool);sm=np.zeros(2,dtype=np.int64);ph=np.eye(1)
    saved=[a.copy() for a in [ag,asp,th,y,pm,sm,ph]]
    def score(*args):calls.append('score');return [1.,2.]
    def fit(*args):
        calls.append('fit')
        return dict(a_primary=ag,a_specific=asp,threshold=th,phi=ph,theta_p_eap=np.zeros((2,1)),theta_p_sd=np.ones((2,1)),category_counts=np.ones((2,2)),loglik_trace=[-3.],n_iter=1,converged=False,termination_reason='max_iter_reached',final_loglik_change=1.,best_start=0,n_parameters=6,primary_identification='correlated')
    def oakes(*args):calls.append('oakes');return dict(labels=['synthetic'],information=[2.],vcov=[.5],se=[np.sqrt(.5)],positive_definite=True,non_pd_reason=None)
    monkeypatch.setattr(sys.modules[module.__package__+'.fitstats'],'_core_module',lambda:SimpleNamespace(two_tier_expected_raw=score,fit_two_tier_grm=fit,two_tier_oakes_se=oakes))
    with pytest.raises(ValueError):
        if route=='score-q':
            holder=SimpleNamespace(a_primary=ag,a_specific=asp,threshold=th,theta_p_eap=np.zeros((2,1)),n_cat=2,n_primary=1,n_specific=1)
            module.expected_raw_two_tier_grm(holder,sm,np.bool_(True))
        elif route=='fit-seed':module.fit_two_tier_grm(y,pm,sm,2,1,1,121,241,1,1e-6,1,np.bool_(True))
        else:module.two_tier_oakes_se(ag,asp,th,ph,y,pm,sm,2,1,1,121,241,np.bool_(True))
    assert calls==[] and unused==[]
    for a,b in zip([ag,asp,th,y,pm,sm,ph],saved):np.testing.assert_array_equal(a,b)
