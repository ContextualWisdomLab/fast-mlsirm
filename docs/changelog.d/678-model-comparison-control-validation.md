# Model-comparison control validation hardening

## Security

- Reject hostile semantic/numeric control objects before caller-defined `__str__`, `__repr__`, or `__float__` callbacks can execute, while preserving accepted relation identities, built-in/NumPy scalar semantics, and Rust-owned model-comparison arithmetic.
