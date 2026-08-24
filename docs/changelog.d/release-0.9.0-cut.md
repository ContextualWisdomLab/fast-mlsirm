# Release cut 0.9.0

## Changed

- Project version is bumped to 0.9.0 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.9.0] - 2026-08-24` release section: new governed contracts (cross-engine
  conformance inventory/provenance/manifest-replay evidence, an external
  validation and transportability profile, a governed structural-model
  pair-decision gate, buyer-facing item-bank lifecycle reports), a new
  Rust-owned crossed/weighted multiple-membership person-effects estimator
  (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001) with CPU-threaded and
  optional GPU kernels, reproducible release-tag-bound PyPI sdist/wheel
  publishing, restriction of production backend selection to Rust-owned
  paths (NumPy parity moved behind an explicit `fit_reference` API), a Rust
  1.97.1 toolchain pin across verification, and a broad continuation of the
  hostile-callback/conversion-protocol hardening sweep across dozens of public
  entry points (CAT, ATA, DIF, equating, scaling, reliability, multilevel,
  response-time, fit-statistics, inference, linking, LLM-judge orchestration,
  parallel-analysis, and rotation/loader concurrency, among others).
- This cut also removes the stale, never-rendered `release-0.8.0-cut.md`
  fragment left over from the abandoned 0.8.0 release attempt (that version
  was never actually tagged or published); its already-recorded
  `[0.8.0] - 2026-08-17` section in `CHANGELOG.md` is left untouched as
  history, and this release supersedes it directly.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
