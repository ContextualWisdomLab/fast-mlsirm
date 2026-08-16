# Rubric integer control callback boundary

## Fixed

- Hardened rubric and blueprint integer normalization so exact built-in integers and genuine supported NumPy integer scalars remain compatible while caller-defined integer/protocol objects are rejected before executable conversion callbacks.
- Preserved the existing score, item-count, replicate-index, seed, and unsigned-64 bounds without changing psychometric arithmetic or Rust numerical ownership.
