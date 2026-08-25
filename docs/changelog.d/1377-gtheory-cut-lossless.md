# Preserve G-theory mastery-cut identity at the Rust boundary

## Fixed

- Reject finite integer and extended-precision mastery cuts when binary64 normalization would change the threshold used by Rust-owned Brennan–Kane `Phi(lambda)` calculations.
- Apply the same lossless cut contract to the direct `phi_lambda()` API and the provenance-safe G-theory pilot handoff, so provenance cannot advertise a threshold that numerical marshalling changes.
- Preserve exactly representable built-in and concrete NumPy integer/floating controls as package-owned built-in `float` values while keeping Boolean, numeric-subclass, protocol-provider, and non-finite controls fail-closed.
- Leave G-study ANOVA/EMS, D-study variance components, Brennan–Kane signal and `Phi(lambda)` arithmetic unchanged and Rust-owned.
