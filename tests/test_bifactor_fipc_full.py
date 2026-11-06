"""One full EM step against an independent reduced-posterior calculation.
Synthetic arithmetic comparison, not study accuracy certification.
"""
import numpy as np
from fast_mlsirm import bifactor_grm as module


def fixture():
    y = np.array([[0,0,0,0],[0,1,1,1],[1,0,0,1],[1,1,1,0],
                  [0,1,0,1],[1,0,1,0],[1,1,0,0],[0,0,1,1]])
    return dict(responses=y, specific_map=np.array([0,0,1,1]), n_cat=2,
                n_specific=2, fixed_a_general=np.array([.7,1.1,.8,1.2]),
                fixed_a_specific=np.array([1.2,-.9,1.1,.6]),
                fixed_threshold=np.array([[.1],[-.3],[.4],[-.2]]),
                initial_mean=np.array([.2,-.4,.6]),initial_sd=np.array([.9,1.2,.8]),
                q_general=121,q_specific=121,max_iter=1,tol=1e-8,device="cpu")


def independent_update(f):
    # Normal GH weights and logistic graded response with two categories.
    x,w=np.polynomial.hermite.hermgauss(121); x=x*np.sqrt(2);w=w/np.sqrt(np.pi)
    nodes=[m+s*x for m,s in zip(f['initial_mean'],f['initial_sd'])]
    sums=np.zeros(3);squares=np.zeros(3)
    for response in f['responses']:
        block_likes=[]
        for s in range(2):
            likelihood=np.ones((121,121))
            for j in range(2*s,2*s+2):
                eta=f['fixed_a_general'][j]*nodes[0][:,None]+f['fixed_a_specific'][j]*nodes[s+1][None,:]+f['fixed_threshold'][j,0]
                p=1/(1+np.exp(-eta));likelihood*=p if response[j] else 1-p
            block_likes.append(likelihood)
        marginal=[b@w for b in block_likes];general=w*marginal[0]*marginal[1];z=general.sum();general/=z
        sums[0]+=general@nodes[0];squares[0]+=general@(nodes[0]**2)
        for s in range(2):
            joint=w[:,None]*block_likes[s]*w[None,:]*marginal[1-s][:,None]/z
            post=joint.sum(axis=0);sums[s+1]+=post@nodes[s+1];squares[s+1]+=post@(nodes[s+1]**2)
    means=sums/len(f['responses']);sd=np.sqrt(squares/len(f['responses'])-means**2)
    return means,sd


def test_full_means_one_update_matches_independent_posterior():
    api=getattr(module,'fit_bifactor_grm_fipc_full',None)
    assert callable(api), "full focal means API absent; zero-mean specific API cannot implement exact model"
    f=fixture();mean,sd=independent_update(f);r=api(**f)
    np.testing.assert_allclose(r.mean,mean,rtol=0,atol=1e-11)
    np.testing.assert_allclose(r.sd,sd,rtol=0,atol=1e-11)
    assert r.n_iter==1 and r.converged is False
    assert r.n_parameters==6
    np.testing.assert_array_equal(r.a_general,f['fixed_a_general'])
    np.testing.assert_array_equal(r.a_specific,f['fixed_a_specific'])
    np.testing.assert_array_equal(r.threshold,f['fixed_threshold'])


def recovery_fixture():
    rng=np.random.default_rng(20260921);N=400;J=12
    ag=np.array([.7,1.1,.8,1.2,.9,1.,.65,1.25,.8,1.05,.9,1.15])
    a_s=np.array([1.2,-.9,1.1,.6,.8,-.7,.9,1.1,.75,.95,-.6,1.05])
    sm=np.repeat(np.arange(3),4);th=np.tile([.7,-.6],(J,1))
    latent=rng.normal(size=(N,4))*[1.1,.8,1.2,.9]+[.3,-.4,.5,.2]
    eta=latent[:,0,None]*ag+latent[:,sm+1]*a_s;u=rng.uniform(size=(N,J))
    y=(u<1/(1+np.exp(-(eta+.7)))).astype(int)+(u<1/(1+np.exp(-(eta-.6)))).astype(int)
    return dict(responses=y,specific_map=sm,n_cat=3,n_specific=3,
        fixed_a_general=ag,fixed_a_specific=a_s,fixed_threshold=th,
        initial_mean=np.array([.4317393112182617,-.3752089975419492,.41162855354038325,.07010105119573212]),
        initial_sd=np.array([1.1967261229047066,.9449176039353109,.9052243466702726,.9512228920116755]),
        q_general=121,q_specific=121,max_iter=1,tol=1e-6)


def test_missing_responses_refused_before_native():
    import pytest
    for bad in [np.nan,-1,np.inf]:
        f=fixture();f['responses']=f['responses'].astype(float);f['responses'][0,0]=bad
        with pytest.raises(ValueError,match='complete observed'):
            module.fit_bifactor_grm_fipc_full(**f)


def test_fractional_map_refused():
    import pytest
    f=fixture();f['specific_map']=np.array([.1,0,1,1])
    with pytest.raises(ValueError,match='specific_map'):
        module.fit_bifactor_grm_fipc_full(**f)


def test_bad_initial_distribution_refused():
    import pytest
    f=fixture();f['initial_sd'][1]=0
    with pytest.raises(ValueError,match='positive'):
        module.fit_bifactor_grm_fipc_full(**f)


def test_quadrature_is_required_without_defaults():
    import inspect
    sig=inspect.signature(module.fit_bifactor_grm_fipc_full)
    for k in ['q_general','q_specific']:
        assert sig.parameters[k].default is inspect.Parameter.empty


def test_native_malformed_shape_is_value_error():
    import pytest
    from fast_mlsirm import _core
    f=fixture()
    with pytest.raises(ValueError,match='y must have length'):
        _core.fit_bifactor_grm_fipc_full(np.array([0],dtype=np.int64),f['specific_map'],8,4,2,2,
            f['fixed_a_general'],f['fixed_a_specific'],f['fixed_threshold'].ravel(),
            f['initial_mean'],f['initial_sd'],121,121,1,1e-8,'cpu')


def test_native_fixed_threshold_rejected():
    import pytest
    f=recovery_fixture();f['fixed_threshold'][0]=[-.6,.7]
    with pytest.raises(ValueError,match='strictly decreasing'):
        module.fit_bifactor_grm_fipc_full(**f,device='cpu')


def test_gpu_false_likelihood_decrease_regression():
    import pytest
    from fast_mlsirm import _core
    # Explicitly hardware-bound: unavailable hardware is NOT acceptance.
    f=recovery_fixture()
    try:
        r=module.fit_bifactor_grm_fipc_full(**f,device='gpu')
    except ValueError as exc:
        if 'GPU-only E-step required' in str(exc):
            pytest.skip('GPU adapter unavailable: hardware evidence unmet')
        raise
    assert r.gpu_execution_used is True and r.cpu_fallback_reason is None
    assert abs(r.loglik_trace[0]-(-4703.858303954978))<1e-9
    assert r.loglik_trace[1]>=r.loglik_trace[0]
    assert r.n_iter==1 and r.converged is False
