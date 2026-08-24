# GPCM admission hardening

## Fixed

- Validate GPCM category, quadrature, iteration, tolerance, integration-point, and RNG-seed controls before caller response materialization, admitting only package-trusted built-in or concrete NumPy scalar identities and passing normalized primitives to Rust.
- Reject complex response evidence before real-valued narrowing and reject infinite response values instead of silently reclassifying them as missing, while preserving ordinary categories plus documented NaN/negative missingness.
- Keep GPCM probabilities, marginal likelihood, estimation, integration, reflection/identification, convergence, and EAP arithmetic unchanged in the Rust numerical core.
