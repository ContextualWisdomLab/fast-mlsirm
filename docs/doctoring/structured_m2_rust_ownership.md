# Structured single-population M2 Rust ownership

## Decision

The public single-population `m2()` path now keeps its numerical owner in
`mlsirm-core` when calibration metadata changes the tangent-space parameter
count. Python validates inputs, marshals arrays, and constructs the result
object; it does not enter `_m2_single_population` or `_projected_m2_numpy` for
production calls.

This slice covers `estimate_population`, `fixed_items`, and `tau_fixed`. The
multigroup and multilevel M2 paths remain a separately tracked migration slice
until equivalent Rust entrypoints own their moment construction and covariance
assembly.

## Scientific boundary

The implementation preserves the existing binary M2 moments, finite-difference
derivatives, complete-case rule, projected quadratic form, RMSEA2 interval,
SRMSR, CFI, and TLIRT calculations. Only the derivative columns representing
estimated or anchored calibration metadata are moved into the existing Rust
implementation. The limited-information fit statistic remains within the
Maydeu-Olivares and Joe (2005, 2006) tangent-space construction, and the
incremental indices remain aligned with the Cai et al. (2023) IRT convention.

## Verification

- `tests/test_fitstats_structured_m2_rust_ownership.py` proves public dispatch
  and fail-closed behavior with a native sentinel.
- Rust unit coverage exercises the structured parameter enumeration and setter
  branches.
- Existing paper-feature fixtures continue to exercise finite M2 output for
  estimated population and anchored-item cases after the PyO3 extension is
  rebuilt.

## APA 7th references

Cai, L., Chung, S. W., & Lee, T. (2023). Incremental model fit assessment in
the case of categorical data: Tucker–Lewis index for item response theory
modeling. *Prevention Science, 24*(3), 455–466.
https://doi.org/10.1007/s11121-021-01253-4

Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information
estimation and goodness-of-fit testing in 2^n contingency tables. *Journal of
the American Statistical Association, 100*(471), 1009–1020.
https://doi.org/10.1198/016214504000002069

Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit
testing in multidimensional contingency tables. *Psychometrika, 71*(4),
713–732. https://doi.org/10.1007/s11336-005-1295-9

International Organization for Standardization. (2023). *ISO/IEC
25010:2023 systems and software engineering—Systems and software quality
requirements and evaluation (SQuaRE)—Product quality model*.
