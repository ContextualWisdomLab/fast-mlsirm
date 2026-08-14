# Second-order diagnostics keep positive-definiteness semantics strict

## Fixed

- Rust-owned observed-information diagnostics now reject negative positive-definiteness tolerances instead of allowing callers to redefine a matrix with small negative eigenvalues as positive definite.
- Zero tolerance remains supported and preserves the strict requirement that every information eigenvalue be positive.
- Oversized second-order matrix dimensions whose square cannot be represented by `usize` now fail closed with a stable package error instead of overflowing dimension arithmetic.
