# Harden S-X² scalar control admission

## Fixed

- Reject caller-defined integer and floating subclasses at the public S-X² control boundary before numeric conversion or compiled-core dispatch, while preserving exact built-in and concrete NumPy scalar compatibility and leaving all S-X²/G², quadrature, and BH/FDR arithmetic Rust-owned.
- Reject built-in or concrete NumPy integer-valued real controls when float64 normalization would change the integer identity, so `min_expected`, `fdr_q`, and `min_effect` cannot be silently rounded before domain validation.
- Reject finite extended-precision NumPy floating controls when conversion to the Rust `f64` boundary would change their value, while preserving exact float16/float32/float64 and lossless long-double controls.
