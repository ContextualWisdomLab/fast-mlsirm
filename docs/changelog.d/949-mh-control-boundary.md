# Mantel-Haenszel control trust boundary

## Fixed

- `mantel_haenszel_dif()` now establishes trusted `fdr_q` and `exclude_studied_item` controls before materializing caller response/group data or discovering the compiled Rust core. Exact built-in bools and supported exact NumPy numeric scalar identities remain compatible; booleans-as-numbers, subclasses, and arbitrary conversion providers fail closed before caller callbacks. Huge exact integers that overflow `float()` raise package `ValueError` rather than a bare `OverflowError`.
- The FDR threshold preserves the existing finite `(0, 1]` domain. The ETS default still includes the studied item in the matching total. Mantel-Haenszel odds ratios, chi-square, ETS delta, standardized P-DIF, A/B/C classes, and BH flags remain Rust-owned.
