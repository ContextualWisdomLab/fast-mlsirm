# GRM response data integrity

## Fixed

- Reject complex-valued graded-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component. Ordinary real/integer categories and missing-value handling retain the existing Rust-owned GRM fitting path.
