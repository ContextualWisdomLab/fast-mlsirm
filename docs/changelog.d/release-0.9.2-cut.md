# Release cut 0.9.2

## Changed

- Project version is bumped to 0.9.2 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.9.2] - 2026-08-27` release section: preregistered external-validation and
  transportability evidence profiles, a domain-neutral Rust/PyO3
  finite-population proportion sampling-design contract (sample size, finite-
  population correction, proportional/equal-cost Neyman allocation, and a
  terminal SRSWOR achieved-proportion artifact with Wang/Konijn exact
  confidence limits) plus a bounded threshold-sorted water-filling capped-strata
  allocation, a domain-neutral lineage-channel-weight evidence contract, a
  shared canonical judge-result IRT projection core, provisional-versus-
  calibrated item-parameter provenance with independent (non-self-referential)
  provenance edges, item-bank lifecycle public-identity replay, customer-facing
  error-copy actionability across the CLI and public API, and a continuation of
  the hostile-input/structural-budget hardening sweep (2PL response/tolerance
  admission, confirmatory loading-pattern evidence, residual interaction-map
  and RSM structural traversal, sampling result-contract replay, release
  acceptance watchdog budget, and release helper import integrity).
- This cut removes the standing predecessor note `release-0.9.1-cut.md`, whose
  substance is permanently recorded in the `[0.9.1] - 2026-08-25` section and
  in git history, mirroring the precedent set by the v0.9.1 cut's removal of
  the stale `release-0.9.0-cut.md` leftover.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
