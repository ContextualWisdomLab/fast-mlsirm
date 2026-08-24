# Nominal-response admission hardening

## Fixed

- Validate nominal category, quadrature, iteration, tolerance, Monte Carlo point, and RNG-seed controls before caller response materialization, accepting only package-trusted built-in or concrete NumPy scalar identities and passing normalized primitives to Rust.
- Reject complex response evidence before real-valued narrowing and reject infinite response values instead of silently reclassifying them as missing, while preserving ordinary real/integer categories plus documented NaN/negative missingness.
- Keep nominal probabilities, marginal likelihood, estimation, integration, convergence, identification, and EAP arithmetic unchanged in the Rust numerical core.
