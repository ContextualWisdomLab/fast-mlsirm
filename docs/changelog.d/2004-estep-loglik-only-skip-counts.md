# Skip bifactor expected-count fill on loglik-only E-step (#2004)

## Changed

- CPU bifactor GRM `e_step` takes `accumulate_counts`;
  [`bifactor_grm_marginal_loglik`](crates/mlsirm-core/src/bifactor_grm.rs)
  passes `false` so the measured posterior/expected-count nest (~2/3 of a
  CP3-shaped profile) is skipped when only the observed-data loglik is
  needed. Fit / Oakes callers still pass `true`. Section timers
  (`enable_estep_nest_profile` / `take_estep_nest_profile`) and the audit
  ledger `docs/perf/estep-nest-audit-2004.md` record the ranking. Defect-A
  `is_obs`/`y` hoist was measured as a regression on `observed = None` and
  was not applied. Pattern-frequency collapse remains #2003 / PR #2005. No
  new numeric defaults (ADR-0028).
