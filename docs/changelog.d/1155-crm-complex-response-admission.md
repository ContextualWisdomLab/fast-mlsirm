# CRM response data integrity

## Fixed

- Reject complex-valued continuous-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component, and reject object-dtype response storage before caller-defined numeric conversion can run. Ordinary real-valued arrays, including `NaN` missing cells, retain the existing Rust-owned CRM fitting path.
