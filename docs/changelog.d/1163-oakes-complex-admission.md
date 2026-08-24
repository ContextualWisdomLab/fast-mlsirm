# Oakes uncertainty input admission

## Fixed

- Reject complex-valued response matrices and factor assignments before any real/integer narrowing can discard imaginary components in the public Oakes standard-error wrapper.
- Preserve existing binary-response missingness and integer factor semantics while keeping Oakes information, finite-difference, inversion, and standard-error arithmetic in the Rust core.
