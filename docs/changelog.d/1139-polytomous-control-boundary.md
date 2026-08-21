# Polytomous fit control-boundary hardening

## Fixed

- Validate the model selector, category count, quadrature rule, iteration cap,
  and tolerance through trusted scalar identities before response
  materialization or Rust-core discovery.
- Reject callback-capable text, numeric subclasses, and lossy numeric
  protocols without changing valid built-in or concrete NumPy scalar behavior.
