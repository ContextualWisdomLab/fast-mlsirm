import sys, json, math, hashlib
sys.path.insert(0, 'tests')
import numpy as np
from fast_mlsirm import fit_two_tier_grm, _core
from fast_mlsirm.regression import absolute_differences
from test_two_tier_focal_gaussian_recovery_native import _continuous_six_latent_fixture

y, ap, _, sm, _, digest = _continuous_six_latent_fixture(64, np.zeros(6), np.ones(6), 20260930)
controls=dict(n_cat=4,n_primary=2,n_specific=4,q_primary=7,q_specific=7,tol=1e-6,n_starts=1,seed=20260930,primary_correlation='identity')
print('responses_sha256',digest,flush=True)
for cap in (1,10,50,100,200,261):
    fits={d:fit_two_tier_grm(y,ap!=0,sm,**controls,max_iter=cap,device=d,gpu_memory_budget_bytes=268435456) for d in ('cpu','gpu')}
    a,b=fits['cpu'],fits['gpu']
    changes={k:absolute_differences(getattr(a,k).ravel(),getattr(b,k).ravel())['max_abs_diff'] for k in ('a_primary','a_specific','threshold','theta_p_eap','theta_p_sd')}
    print(json.dumps(dict(cap=cap,cpu_n_iter=a.n_iter,gpu_n_iter=b.n_iter,initial_ll_delta=b.loglik_trace[0]-a.loglik_trace[0],final_ll_delta=b.loglik_trace[-1]-a.loglik_trace[-1],max_delta=changes,adapter=b.gpu_adapter_name,adapter_backend=b.gpu_adapter_backend,item13_cpu=[*a.a_primary[13],a.a_specific[13],*a.threshold[13]],item13_gpu=[*b.a_primary[13],b.a_specific[13],*b.threshold[13]])),flush=True)
    if cap==261:
        # Independent synthetic-only category probabilities at the same q7
        # primary/specific nodes: diagnostic oracle, not a production owner.
        nodes=np.polynomial.hermite.hermgauss(7)[0]*math.sqrt(2)
        def logistic(x):
            if x>=0: return 1/(1+math.exp(-x))
            z=math.exp(x);return z/(1+z)
        def probs(fit,g,s):
            eta=fit.a_primary[13,0]*g+fit.a_specific[13]*s
            cum=[1.,*[logistic(eta+d) for d in fit.threshold[13]],0.]
            return [cum[k]-cum[k+1] for k in range(4)]
        gap=max(abs(x-z) for g in nodes for s in nodes for x,z in zip(probs(a,g,s),probs(b,g,s)))
        print('item13_same_node_category_probability_max_delta',gap,flush=True)
