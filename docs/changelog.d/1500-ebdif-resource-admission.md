# Empirical Bayes DIF item-evidence admission

## Fixed

- Bound each public Empirical Bayes Mantel-Haenszel `mh` and `se` vector to 20,000,000 item entries before package-owned contiguous `float64` allocation.
- Use exact NumPy shape metadata or exact built-in list/tuple length before scalar materialization, preserving callback-free carrier admission and existing complex/numeric diagnostics.
- Reject unequal `mh`/`se` lengths and the fewer-than-two-item domain from callback-free carrier metadata before value-wise validation or dense `float64` conversion, while preserving resource and malformed-carrier error precedence.
- Require admitted integer and wider NumPy floating item evidence to preserve its exact finite value across Rust `f64` normalization; lossy or overflowed evidence now fails before compiled-core discovery rather than silently changing the MH statistic or standard error.
- Seal the concrete Rust result envelope before public marshalling: require the exact seven-key mapping, exact finite Python-float scalars, exact one-dimensional C-contiguous NumPy `float64` vectors matching the current `PyArray1::from_slice` binding layout with item-bound cardinalities, and flattened five-category evidence of exactly `n_items * 5` finite values before reshape. Foreign mapping/scalar/array conversion protocols and strided stale-native views fail closed.
- Replay deterministic Rust-owned output domains before public marshalling: `tau2` must exactly equal positive `tau2_raw` or zero when `tau2_raw <= 0`, posterior variances must be non-negative, and shrinkage weights and ETS category probabilities must remain in `[0, 1]`; finite negative `tau2_raw` remains valid as the documented pre-floor diagnostic.
- Keep prior estimation, the variance-floor calculation itself, shrinkage weights, posterior means/variances, and ETS category probabilities in the Rust numerical core; these changes are limited to Python validation and marshalling.
