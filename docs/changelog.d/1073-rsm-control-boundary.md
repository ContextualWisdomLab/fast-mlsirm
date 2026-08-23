# RSM control and response admission safety

## Fixed

- Harden Rating Scale Model semantic-control admission so `n_cat`, quadrature size, iteration limits, and tolerance are normalized from trusted scalar identities before caller data or Rust capability work; hostile scalar subclasses and conversion/hash providers are rejected without callback dispatch while established built-in and NumPy compatibility remains unchanged.
- Reject arbitrary caller response array providers and container/numeric subclasses before NumPy protocol execution, while preserving exact NumPy arrays, ordinary built-in rows, and exact NumPy row arrays containing supported real numeric evidence.
- Reject complex, object, and textual Rating Scale Model response storage before real-valued narrowing or caller element conversion, then marshal only admitted Boolean/integer/real numeric evidence to contiguous `float64` while preserving `NaN` missingness and the Rust-owned Andrich calibration semantics.
