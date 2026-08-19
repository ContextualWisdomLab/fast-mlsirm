# Harden S-X² scalar control admission

## Fixed

- Reject caller-defined integer and floating subclasses at the public S-X² control boundary before numeric conversion or compiled-core dispatch, while preserving exact built-in and concrete NumPy scalar compatibility and leaving all S-X²/G², quadrature, and BH/FDR arithmetic Rust-owned.
