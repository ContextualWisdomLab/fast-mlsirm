# Delta-plot control trust boundary

## Fixed

- `delta_plot()` now establishes trusted selector, scalar, range, and iteration controls before materializing caller response/group data or discovering the compiled Rust core. Exact built-in strings and supported exact NumPy numeric scalar identities remain compatible; booleans, subclasses, unused-branch hostiles, and arbitrary conversion providers fail closed before caller callbacks. Huge exact integers that overflow `float()` raise package `ValueError` rather than a bare `OverflowError`.
- Normal-threshold `alpha` preserves the Rust `(0, 1)` domain, constraint ranges preserve `0 <= lo < hi <= 1`, fixed thresholds must be finite, additive adjustment counts stay positive, and `max_iter` is bounded by the package-wide `MAX_MAX_ITER` ceiling. Angoff Delta plot proportions, transforms, purification, thresholds, DIF flags, and result arithmetic remain Rust-owned.
