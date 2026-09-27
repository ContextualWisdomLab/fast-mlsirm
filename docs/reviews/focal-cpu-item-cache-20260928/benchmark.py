"""Synthetic CPU scoring cost, not study estimates.
Cai (2010), pp.608-609: fixed item/node probabilities do not depend on person.
Timing uses Python3.12 time.perf_counter; SHA256 binds actual library outputs.
https://docs.python.org/3.12/library/time.html#time.perf_counter
"""
import json,sys,time,hashlib,statistics
from pathlib import Path
import numpy as np
from fast_mlsirm import score_two_tier_grm_orthogonal
f=json.loads(Path("tests/fixtures/two_tier_focal/six_latent_same_node.json").read_text())
y=np.array(f["responses"],dtype=float).reshape(3,16)
y[~np.array(f["observed"],dtype=bool).reshape(3,16)]=np.nan
y=np.tile(y,(128,1))
k=dict(responses=y,primary_map=np.array(f["primary_map"],dtype=bool).reshape(16,2),specific_map=np.array(f["specific_map"],dtype=np.int64),a_primary=np.array(f["a_primary"]).reshape(16,2),a_specific=np.array(f["a_specific"]),threshold=np.array(f["threshold"]).reshape(16,3),latent_mean=np.array(f["latent_mean"]),latent_sd=np.array(f["latent_sd"]),n_cat=4,n_primary=2,n_specific=4,q_primary=5,q_specific=5)
if "--cache" in sys.argv:k["cache_item_tables"]=True
t=[]
for _ in range(3):
 start=time.perf_counter();r=score_two_tier_grm_orthogonal(**k);t.append(time.perf_counter()-start)
h=hashlib.sha256()
for a in (r.mean,r.second,r.sd):h.update(a.tobytes())
h.update(r.loglik.hex().encode())
print(json.dumps(dict(n_persons=len(y),q_primary=5,q_specific=5,elapsed_seconds=t,median_seconds=statistics.median(t),output_sha256=h.hexdigest(),cache_item_tables="--cache" in sys.argv)))
