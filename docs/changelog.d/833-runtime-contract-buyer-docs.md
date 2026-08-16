# Runtime contract buyer-facing ownership

## Fixed

- Locked the Claude runtime-contract TOML block to package metadata and
  Rust-required `auto` ownership, and removed the stale buyer-facing claim that
  `auto` selected NumPy when the compiled core is missing. README (including
  the CLI examples), `fast-mlsirm fit --help`, `FitConfig` comments, commercial
  Operational Notes, PRD, TRD, and ADR-0002 now tell a purchaser to install the
  Rust extension or pass explicit `backend="numpy"` for parity testing. The
  auto fail-closed error now names that next action without reflecting local
  paths or ABI details.
