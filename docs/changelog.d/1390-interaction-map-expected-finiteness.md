# Interaction-map expected-evidence finiteness

## Fixed

- Preserve `NaN` exclusively as an observed-response missingness marker while rejecting `NaN` and infinity in fitted model expectations before complete-case filtering or Rust factorization.
- Replay the same expected-evidence finiteness contract in the Rust core so direct PyO3/core callers cannot silently turn invalid model expectations into missing cells that change the analyzed interaction rectangle.

Gabriel factorization, singular values, coordinates, reconstruction, unexplained residuals, distance, cross-term arithmetic, and observed-response missingness remain unchanged and Rust-owned.
