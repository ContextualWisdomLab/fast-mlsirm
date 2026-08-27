# Empirical Bayes DIF item-evidence resource bound

## Fixed

- Bound each public Empirical Bayes Mantel-Haenszel `mh` and `se` vector to 20,000,000 item entries before package-owned contiguous `float64` allocation.
- Use exact NumPy shape metadata or exact built-in list/tuple length before scalar materialization, preserving callback-free carrier admission and existing complex/numeric diagnostics.
- Keep prior estimation, shrinkage weights, posterior means/variances, and ETS category probabilities in the Rust numerical core; this change is limited to Python validation and marshalling.
