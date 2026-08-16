### Security / trust boundary

- `delta_plot()` now validates and normalizes `threshold`, `alpha`, `fixed_threshold`, `extreme`, `const_range`, `nr_add`, `purify`, and `max_iter` before caller data materialization or compiled-core discovery. Exact built-in strings and supported NumPy scalar identities remain compatible; booleans, subclasses, and arbitrary conversion providers are rejected without executing caller callbacks.
- `max_iter` is bounded by the package-wide `MAX_MAX_ITER` resource ceiling. Angoff Delta-plot proportions, delta transforms, major-axis fit, purification, thresholds, DIF flags, and result statistics remain Rust-owned.
