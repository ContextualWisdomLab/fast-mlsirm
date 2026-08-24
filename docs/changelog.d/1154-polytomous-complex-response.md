# Complex-valued polytomous response admission

## Fixed

- Reject complex-valued polytomous response matrices before any `float64` narrowing can discard imaginary components and turn a different observed category into a valid-looking real category.
- Preserve real integer categories plus `NaN` and `-1` missingness semantics across calibration, scoring, DIF, item/person fit, and other callers of the shared response-admission boundary without changing Rust-owned psychometric arithmetic.
