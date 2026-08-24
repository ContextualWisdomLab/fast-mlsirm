# Selection utility numeric trust boundary

## Fixed

- Hardened classical selection-utility and Taylor-Russell scalar controls so booleans, non-real objects, and non-finite values fail with package-owned validation before compiled Rust discovery.
- Prevented arbitrary caller-defined `__float__` callbacks from executing during public control marshalling while preserving genuine Python/NumPy real scalar compatibility and keeping all BCG, Naylor-Shine, and Taylor-Russell arithmetic Rust-owned.
- Normalized exact built-in integers outside the representable float range to the same package-owned validation error instead of leaking `OverflowError`.
