# Runtime contract buyer-facing ownership

## Fixed

- Locked the Claude runtime-contract TOML block to package metadata and
  Rust-required `auto` ownership, and removed the stale buyer-facing claim that
  `auto` transparently falls back to NumPy when the compiled core is missing.
  README, commercial-readiness, PRD, TRD, and ADR-0002 now tell a purchaser to
  install the Rust extension or pass explicit `backend="numpy"` for parity
  testing.
