# Bound RSM responses before dense materialization

## Fixed

- Reject Rating Scale Model response evidence above 20,000,000 logical cells during the callback-free source preflight, including oversized exact NumPy arrays and exact NumPy rows nested in trusted built-in sequences, before NumPy stacking or contiguous `float64` allocation while preserving existing response semantics and Rust-owned Andrich arithmetic.
