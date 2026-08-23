# Bound G-theory score evidence before dense materialization

## Fixed

- `gtheory_pi()` and `phi_lambda()` now reject score evidence outside the documented two-dimensional persons-by-items shape before dense NumPy materialization; `gtheory_pio()` applies the same fail-first contract to its three-dimensional persons-by-items-by-occasions shape.
- G-theory score evidence now has an explicit 20,000,000-cell logical-resource ceiling that applies to exact NumPy views and trusted built-in sequence trees before a contiguous `float64` copy is allocated.
- Existing exact NumPy arrays, ordinary built-in list/tuple score trees, exact NumPy-array rows, callback-free cycle rejection, and Rust-owned G-study/D-study/`Phi(lambda)` arithmetic remain unchanged.