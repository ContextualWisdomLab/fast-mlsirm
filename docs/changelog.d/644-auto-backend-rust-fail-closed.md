# Automatic backend preserves Rust numerical ownership

## Changed

- Changed `backend="auto"` so a missing compiled Rust core fails closed instead of silently selecting the independent NumPy reference implementation.
- Kept explicit `backend="numpy"` as an explicit reference/parity choice while preserving automatic Rust resolution and Rust CPU/GPU device fallback semantics.
