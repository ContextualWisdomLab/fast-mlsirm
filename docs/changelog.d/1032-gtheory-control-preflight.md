# Validate G-theory controls before data and Rust discovery

## Fixed

- G-theory D-study sizes and `Phi(lambda)` scalar controls now fail closed before caller-owned score-array materialization and before compiled Rust capability discovery when invalid, while preserving the existing callback-free Python/NumPy scalar contract and unchanged Rust-owned psychometric arithmetic.
