# Fit-statistics infit/outfit and M2 fail closed

## Fixed

- Public `infit_outfit()` and ordinary `m2()` fail closed when the compiled Rust
  core or required entrypoints are missing, completing the residual ownership
  gaps from issue #627 after S-X² and person-fit hardening.
