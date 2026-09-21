# Baseline reproducer: released fast_mlsirm 0.11.4 (no AC changes), MML only.
import numpy as np, fast_mlsirm, sys
from fast_mlsirm.bifactor_multigroup import fit_bifactor_grm_multigroup
rows = [[0,1,2,0],[1,2,0,1],[2,0,1,2],[0,2,1,0],[1,1,1,1],[2,2,2,2],
        [0,0,0,0],[2,1,0,2],[1,0,2,1],[0,2,0,1],[1,2,1,0],[2,0,2,1]]
y = np.array(rows); group = np.arange(12) % 2; smap = np.array([0,0,1,1])
print(sys.executable, fast_mlsirm.__version__, fast_mlsirm.__file__)
for label, anchor in [("anchor=None", None), ("anchor=[T,T,F,F]", np.array([True,True,False,False]))]:
    try:
        fit = fit_bifactor_grm_multigroup(y, group, smap, 3, 2, anchor, q_general=7, q_specific=7,
            max_iter=60, tol=1e-4, n_starts=1, seed=42, device="cpu")
        print(label, "OK", fit.termination_reason, fit.n_iter)
    except Exception as e:
        print(label, type(e).__name__, e)
