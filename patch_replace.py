import re

with open("python/fast_mlsirm/estimators/marginal.py", "r") as f:
    text = f.read()

# Pattern 1
text = re.sub(
    r'q = float\(np.sum\(r_i \* _log_sigmoid\(eta\) \+ \(n_i - r_i\) \* _log_sigmoid\(-eta\)\)\)',
    r'q = float(np.vdot(r_i, eta) + np.vdot(n_i, _log_sigmoid(-eta)))',
    text
)

# Pattern 2
text = re.sub(
    r'dist = np.sqrt\(eps_distance \+ np.sum\(diff \* diff, axis=1\)\)',
    r'dist = np.sqrt(eps_distance + np.einsum("ij,ij->i", diff, diff))',
    text
)

# Pattern 3
# There are two instances that look exactly like this, they were formatted differently due to ruff
text = re.sub(
    r'np.sum\(\s*rbar \* _log_sigmoid\(e\) \+ \(n_all - rbar\) \* _log_sigmoid\(-e\)\s*\)',
    r'np.vdot(rbar, e) + np.vdot(n_all, _log_sigmoid(-e))',
    text
)

with open("python/fast_mlsirm/estimators/marginal.py", "w") as f:
    f.write(text)
