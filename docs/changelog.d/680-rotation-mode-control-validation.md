# Rotation mode validation hardening

## Security

- Reject non-built-in-string rotation `mode` controls before caller-defined representation callbacks can execute, while preserving the existing orthogonal/oblique vocabulary, aliases, default resolution, and Rust-owned rotation numerics.
