# Second-order diagnostics keep positive-definiteness semantics strict

## Fixed

- Rust-owned observed-information diagnostics now reject negative positive-definiteness tolerances instead of allowing callers to redefine a matrix with small negative eigenvalues as positive definite.
- Zero tolerance remains supported and preserves the strict requirement that every information eigenvalue be positive.
