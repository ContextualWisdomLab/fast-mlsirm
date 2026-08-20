# RSM control callback safety

## Fixed

- Harden Rating Scale Model semantic-control admission so `n_cat`, quadrature size, iteration limits, and tolerance are normalized from trusted scalar identities before caller data or Rust capability work; hostile scalar subclasses and conversion/hash providers are rejected without callback dispatch while established built-in and NumPy compatibility remains unchanged.
