# Release cut 0.11.1

## Changed

- Project version is bumped to 0.11.1 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.11.1] - 2026-09-18` release section, headlined by the bifactor GRM
  dense-quadrature EM stall fix (#1976 / #1981): zero-weight Gauss–Hermite
  nodes are skipped in the E-step, and a bit-identical start-slope plateau is
  reported as `numerical_em_stall` instead of a false `tolerance_met`. Also
  folded from the post-0.11.0 deferred lineage: ADR-0028 naming/defaults
  application for `dif`/`deltaplot`/`polytomous` (#1962), bifactor modules
  (#1963/#1969), and core IRT fitters (#1964/#1973).
- This cut removes the standing predecessor note `release-0.11.0-cut.md`, whose
  substance is permanently recorded in the `[0.11.0] - 2026-09-17` section and
  in git history.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
