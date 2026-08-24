# Runtime contract buyer-facing ownership

## Fixed

- Locked the Claude runtime-contract TOML block to package metadata and
  Rust-required `auto` ownership, and removed the stale buyer-facing claim that
  `auto` selected NumPy when the compiled core is missing. README, `FitConfig`
  comments, commercial Operational Notes, the buyer demo storyboard, sales
  `--check-import` help, PRD, TRD, and ADR-0002 now tell purchasers to install
  the Rust extension for production fitting. Explicit parity/reference work uses
  `fast-mlsirm fit --reference` at the CLI and the `fast_mlsirm.fit_reference`
  API in Python; direct production `fast_mlsirm.fit(...)` does not accept NumPy
  as a production backend. Release acceptance now rejects a NumPy outcome on
  `fit --backend auto`. The auto fail-closed error names the Python reference
  API without reflecting local paths or ABI details.
