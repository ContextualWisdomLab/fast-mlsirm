# #2093 root-cause probe on released 0.11.4 (isolated venv). Toy fixture only; no research data.
import numpy as np, fast_mlsirm
from fast_mlsirm.bifactor_multigroup import fit_bifactor_grm_multigroup
rows=[[0,1,2,0],[1,2,0,1],[2,0,1,2],[0,2,1,0],[1,1,1,1],[2,2,2,2],[0,0,0,0],[2,1,0,2],[1,0,2,1],[0,2,0,1],[1,2,1,0],[2,0,2,1]]
y=np.array(rows); smap=np.array([0,0,1,1])
print("fast_mlsirm", fast_mlsirm.__version__)
def run(label, groups, anchor=None, q=7, max_iter=60, seed=42):
    try:
        f=fit_bifactor_grm_multigroup(y,groups,smap,3,2,anchor,q_general=q,q_specific=q,max_iter=max_iter,tol=1e-4,n_starts=1,seed=seed,device="cpu")
        tr=np.asarray(f.loglik_trace); d=np.diff(tr)
        print(f"{label:34s} OK   n_iter={f.n_iter:3d} min_delta={d.min() if d.size else 0:+.3e} sd_g1={f.general_sd[-1]:.4f} mean_g1={f.general_mean[-1]:+.4f}")
    except ValueError as e:
        print(f"{label:34s} FAIL {str(e).split('first error: ')[-1][:90]}")
g2=np.arange(12)%2
run("baseline 2grp anchor=None q7", g2)
for q in (15,31,61): run(f"2grp anchor=None q{q}", g2, q=q)
run("1grp (delegates to single-group)", np.zeros(12,int))
run("2grp anchor=None seed=7", g2, seed=7)
# Trace up to the failing iteration to see what moves
for mi in (7,8,9):
    try:
        f=fit_bifactor_grm_multigroup(y,g2,smap,3,2,None,q_general=7,q_specific=7,max_iter=mi,tol=1e-12,n_starts=1,seed=42,device="cpu")
        print(f"max_iter={mi}: ll_last={f.loglik_trace[-1]:.9f} mean_g1={f.general_mean[1]:+.6f} sd_g1={f.general_sd[1]:.6f} a_g={np.round(f.a_general[0],4)}")
    except ValueError as e: print(f"max_iter={mi}: FAIL {str(e)[-80:]}")
