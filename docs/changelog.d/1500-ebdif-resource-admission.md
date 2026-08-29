# Empirical Bayes DIF item-evidence admission

## Fixed

- Bound each public Empirical Bayes Mantel-Haenszel `mh` and `se` vector to 20,000,000 item entries before package-owned contiguous `float64` allocation.
- Use exact NumPy shape metadata or exact built-in list/tuple length before scalar materialization, preserving callback-free carrier admission and existing complex/numeric diagnostics.
- Reject unequal `mh`/`se` lengths and the fewer-than-two-item domain from callback-free carrier metadata before value-wise validation or dense `float64` conversion, while preserving resource and malformed-carrier error precedence.
- Require admitted integer and wider NumPy floating item evidence to preserve its exact finite value across Rust `f64` normalization; lossy or overflowed evidence now fails before compiled-core discovery rather than silently changing the MH statistic or standard error.
- Seal the concrete Rust result envelope before public marshalling: require the exact seven-key mapping, exact finite Python-float scalars, exact one-dimensional NumPy `float64` vectors with item-bound cardinalities, and flattened five-category evidence of exactly `n_items * 5` finite values before reshape. Foreign mapping/scalar/array conversion protocols fail closed without callbacks.
- Keep prior estimation, shrinkage weights, posterior means/variances, and ETS category probabilities in the Rust numerical core; these changes are limited to Python validation and marshalling.
