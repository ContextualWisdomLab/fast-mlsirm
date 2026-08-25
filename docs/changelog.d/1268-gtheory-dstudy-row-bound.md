# Bound G-theory D-study result-row requests

## Fixed

- `gtheory_pi()`, `gtheory_pio()`, and `phi_lambda()` now reject D-study request vectors above 10,000 rows before score materialization or compiled-core discovery.
- D-study result-row count is bounded independently from the existing 1,000,000 per-prime magnitude ceiling, so small valid prime values cannot be repeated to request an unbounded native result table.
- Exact built-in list/tuple controls, trusted Python/NumPy integer entries, the existing per-prime size bound, and all Rust-owned G-study/D-study/`Phi(lambda)` arithmetic remain unchanged.
