# Check where dist is calculated using 3D broadcasting in python/fast_mlsirm/estimators/marginal.py
grep -n "eps_distance + np.sum(diff \* diff" python/fast_mlsirm/estimators/marginal.py
