# Bound CDM evidence before dense materialization

## Fixed

- Reject response and Q-matrix evidence above 20,000,000 logical cells during callback-free preflight, including oversized exact NumPy leaves nested in trusted built-in sequences, before NumPy materialization or `float64` allocation while preserving existing valid evidence and Rust-owned CDM arithmetic.
