# Empirical Bayes DIF item-evidence admission

## Fixed

- Bound each public Empirical Bayes Mantel-Haenszel `mh` and `se` vector to 20,000,000 item entries before package-owned contiguous `float64` allocation.
- Use exact NumPy shape metadata or exact built-in list/tuple length before scalar materialization, preserving callback-free carrier admission and existing complex/numeric diagnostics.
- Require admitted integer and wider NumPy floating item evidence to preserve its exact finite value across Rust `f64` normalization; lossy or overflowed evidence now fails before compiled-core discovery rather than silently changing the MH statistic or standard error.
- Keep prior estimation, shrinkage weights, posterior means/variances, and ETS category probabilities in the Rust numerical core; these changes are limited to Python validation and marshalling.
