# CAT administration data integrity

## Fixed

- Reject administered item indices that cannot be represented losslessly as signed 64-bit identities before range/mask handling, and reject complex-valued binary responses before any real-valued coercion can discard their imaginary component. Ordinary signed indices and real 0/1 responses retain the existing Rust-owned CAT likelihood, ability-estimation, and information paths.
