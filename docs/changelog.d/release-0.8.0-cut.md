# Release cut 0.8.0

## Changed

- Project version is bumped to 0.8.0 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.8.0] - 2026-08-17` release section: governed contract additions
  (multilevel/multiple-membership/longitudinal design, structural
  model-relation and leakage-safe model-validation units, post-pilot
  item-bank lifecycle, RAG scoring/perturbation-anchor/facets-calibration
  adapters, enterprise issue-intelligence observation/calibration/reporting
  contracts, essay facets/score/validation HTML reports, paired rating-range
  and essay-facets synthetic recovery evidence), a broad Rust-ownership
  hardening sweep across dozens of public entry points (CAT, ATA, DIF,
  equating, scaling, reliability, multilevel, response-time, fit-statistics,
  inference, linking, LLM-judge orchestration, parallel-analysis, plausible
  values, Rasch-CML, model-comparison, and rotation/loader concurrency) that
  reject hostile Python callback/conversion-protocol inputs before native
  dispatch, and accessibility/documentation polish (exact-value tooltips,
  tabular numerals, print styles, row headers, architecture baseline,
  Python 3.14 CI).
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
