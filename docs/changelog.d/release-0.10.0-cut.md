# Release cut 0.10.0

## Changed

- Project version is bumped to 0.10.0 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.10.0] - 2026-09-17` release section: two-stage polytomous bifactor GRM
  calibration (single-group and multiple-group concurrent, #1912), expected-
  total-score and focal-dimension monotonicity diagnostics for unidimensional,
  multidimensional, and bifactor graded fits (#1873, #1888, #1889, #1928),
  purified polytomous DIF with per-item and per-focal-group anchor sets
  (#1874, #1890, #1891), retirement of `logistic_dif`'s `jg_class` to "not
  applicable" (a breaking change, #1880), required (no unsourced-default)
  quadrature node-count arguments across the new bifactor/monotonicity APIs (a
  breaking change, #1933), symmetric (sign-preserving) slope-magnitude bounds
  for `fit_mmle_2pl`, `fit_testlet`, and `fit_mixture` that stop silently
  floor-clamping reverse-keyed slopes to a near-zero value (breaking changes,
  #1884, #1885), an `at_bound` boundary-value report for `fit_mixed_items`
  estimates (#1882), reverse-keyed and reflection-classification contract
  coverage (#1870, #1883), a fail-closed assertion for dedicated Statistical
  Studies recovery jobs (#1937), and this release cycle's own doc/changelog
  self-corrections (#1938, #1940).
- Eleven of the PRs folded into this release (#1926, #1888, #1889, #1928,
  #1890, #1891, #1870, #1882, #1883, #1884, #1933) had originally merged with
  no `docs/changelog.d` fragment; their fragments were written retroactively
  for this cut from each PR's title, body, and linked issue, sourced against
  the actual merged public API surface rather than restated from memory.
- This cut also removes the standing predecessor note `release-0.9.1-cut.md`,
  whose substance is permanently recorded in the `[0.9.1] - 2026-08-25`
  section and in git history, mirroring the precedent set by that cut's own
  removal of `release-0.9.0-cut.md`.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
