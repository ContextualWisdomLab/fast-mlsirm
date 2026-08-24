# Mixed-format response admission

## Fixed

- Reject complex-valued mixed-format response evidence before real-valued marshalling so imaginary components cannot be silently discarded before categorical validation and Rust-owned calibration.
