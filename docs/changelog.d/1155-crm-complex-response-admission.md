# CRM response data integrity

## Fixed

- Reject complex-valued continuous-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component. Ordinary real-valued arrays, including `NaN` missing cells, retain the existing Rust-owned CRM fitting path.
