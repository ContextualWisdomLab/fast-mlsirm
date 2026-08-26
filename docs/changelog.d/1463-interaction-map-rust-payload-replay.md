# Interaction-map Rust payload replay

## Fixed

- Validate the Rust-owned residual interaction-map payload before Python coercion or NumPy marshalling: counts, retained identifiers, original indices, extrema identities, vector lengths, and finite numerical scalars must match the validated v1 input/result contract. Stale or foreign native bindings now fail closed with package-owned errors instead of invoking coercion hooks or exposing structurally inconsistent payloads.
