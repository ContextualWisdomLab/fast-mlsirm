# Fleiss kappa control trust boundary

## Fixed

- Hardened the public Fleiss/Conger kappa control boundary so explicit category counts and exact-mode selection are validated without executing caller-defined integer, index, or truthiness callbacks before ratings materialization or compiled-core discovery.
- Preserved genuine Python/NumPy scalar compatibility, capped explicit and inferred category counts at the Rust contract maximum of 10,000, and kept all agreement arithmetic Rust-owned.
