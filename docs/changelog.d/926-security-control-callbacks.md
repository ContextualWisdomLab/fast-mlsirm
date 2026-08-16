### Changed

- Harden answer-copying scalar controls so `wollack_omega(..., n_options=...)`, `k_index(..., copier, source)`, and `k_variants(..., copier, source)` reject caller-defined Python or NumPy integer subclasses before comparison, conversion, or native-core discovery. Exact supported NumPy integer scalars remain accepted; all probability, tail, regression, and answer-copying arithmetic remains Rust-owned and unchanged.
