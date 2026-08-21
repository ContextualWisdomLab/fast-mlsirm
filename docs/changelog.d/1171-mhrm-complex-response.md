# Reject lossy complex MH-RM responses

## Fixed

- Reject complex-valued MH-RM response matrices before real-valued narrowing can discard imaginary response evidence, while preserving existing real binary/GPCM response and Rust-owned estimator behavior.
