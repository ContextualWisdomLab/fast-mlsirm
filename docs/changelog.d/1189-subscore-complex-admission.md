# Subscore complex-evidence admission

## Fixed

- Reject complex-valued response and subscale-assignment evidence before real-valued marshalling so imaginary components cannot be silently discarded before Rust-owned Haberman subscore analysis.
