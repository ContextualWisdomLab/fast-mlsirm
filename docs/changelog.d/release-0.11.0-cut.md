# Release cut 0.11.0

## Changed

- Project version is bumped to 0.11.0 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.11.0] - 2026-09-17` release section, headlined by Rust-owned OLS with
  HC0–HC3 sandwich covariance (`fit_ols_hc`, `contrast`, and χ²/F/t helpers;
  #1982). Also folded from the post-0.10.0 lineage through the OLS merge tip:
  bifactor GRM Oakes standard errors, GPU-parallel bifactor E-step / joint
  person bootstrap / Lord–Wingersky recursion (#1912 stages), FIPC polytomous
  bifactor calibration, two-tier GRM stage 4, required polytomous DIF controls
  (#1958), purified logistic DIF `flagged_bh` clarification (#1941), stage-1
  mirt fixture category-order fix (#1950), fail-closed pytest skip/xfail
  outcomes (#1732/#1936), clippy lint triage (#1905), graphify upstream fixes
  (#1847/#1833), Bock–Zimowski locator verification notes (#1927), and
  ADR-0028 public API naming / unsourced-defaults policy docs and inventory
  tooling (#1959/#1961). Intentionally deferred to the next release: bifactor
  ADR-0028 rename/defaults application (#1963/#1969) and bifactor stall fix
  #1981.
- This cut removes the standing predecessor note `release-0.10.0-cut.md`, whose
  substance is permanently recorded in the `[0.10.0] - 2026-09-17` section and
  in git history.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
