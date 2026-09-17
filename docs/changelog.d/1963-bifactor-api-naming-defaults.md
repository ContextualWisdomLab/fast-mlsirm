# ADR-0028 naming/defaults applied to the bifactor modules (#1963)

## Deprecated

- **`fast_mlsirm.bifactor_recursion.bifactor_lord_wingersky` is renamed to
  `enumerate_bifactor_lord_wingersky`.** The old name remains available for
  one minor release and emits `DeprecationWarning`; it forwards to the new
  name with identical behavior.
- **`fast_mlsirm.bifactor_recursion.direct_enumeration_bifactor` is renamed to
  `enumerate_bifactor_direct`.** Same one-minor-release deprecated-alias
  treatment as above.
- **`fast_mlsirm.bifactor_scoreability.bifactor_scoreability` is renamed to
  `assess_bifactor_scoreability`.** Same one-minor-release deprecated-alias
  treatment; both old and new names are exported from `fast_mlsirm` and
  `fast_mlsirm.bifactor_scoreability`.
- **`fast_mlsirm.bifactor_scoreability.bifactor_scoreability_from_logit_slopes`
  is renamed to `assess_bifactor_scoreability_from_logit_slopes`.** Same
  one-minor-release deprecated-alias treatment.

## Changed

- **`fit_bifactor_grm` (`bifactor_grm.py`) no longer defaults `max_iter`,
  `tol`, `n_starts`, or `seed`.** All four are now required caller
  arguments: `max_iter`/`tol` are unsourced iteration/convergence precision
  controls, `n_starts` is an unsourced replicate count, and `seed` is a
  fixed stochastic seed baked into the previous default — the same ADR-0028
  reasoning already applied to `dif_polytomous_purified` (#1958/#1960) and
  the `q_general`/`q_specific` node counts (#1929) on this same function.
- **`fit_bifactor_grm_fipc` (`bifactor_grm.py`) no longer defaults `max_iter`
  or `tol`.** Same reasoning as above; `q_general`, `q_specific`,
  `newton_iter`, `ridge`, and `estimate_specific_vars` are unchanged
  (out of ADR-0028 policy scope per the decision table).
- **`fit_bifactor_grm_multigroup` (`bifactor_multigroup.py`) no longer
  defaults `max_iter`, `tol`, `n_starts`, or `seed`.** Same reasoning as
  `fit_bifactor_grm`.
- **`run_bifactor_bootstrap` (`bifactor_bootstrap.py`) no longer defaults
  `base_seed`, `ci_level`, `max_iter`, `n_starts`, or `tol`.** `base_seed` is
  a fixed stochastic seed; `ci_level` is an unsourced decision threshold;
  `max_iter`/`tol` are unsourced convergence controls; `n_starts` is an
  unsourced replicate count. These five parameters are now required and
  keyword-only (a `*` was added ahead of them to keep the remaining
  optional parameters — `group_ids`, `n_groups`, `anchor_mask`, `n_jobs`,
  `device`, `estimate_specific_vars` — keyword-only-compatible without
  reordering the required, non-defaulted set ahead of them positionally).
- **`assess_bifactor_scoreability` / `assess_bifactor_scoreability_from_logit_slopes`
  no longer default `zero_tolerance`.** `0.0` was an unsourced
  decision-threshold flag cutoff; it is now a required keyword-only
  argument on both the new names and their deprecated aliases.
- No "keep" decision in this issue's slice of
  `docs/api/renames-and-defaults-20260917.csv` (device, `q_general`,
  `q_specific`, `newton_iter`, `ridge`, `general_factor`, `n_groups`,
  `n_jobs`, `group_ids`, `anchor`/`anchor_mask`,
  `estimate_specific_vars`) carries a cited APA 7th source in its
  `default_rationale` column, so no docstring citations were added for
  this issue; the existing Golub & Welsch (1969) / Gibbons et al. (2007) /
  Cai, Yang & Hansen (2011) / Andrews & Buchinsky (2000) references on
  these five modules are unchanged.
- **Not yet mirrored on the PyO3 side (Python-only change; follow-up
  needed).** The compiled `crates/fast-mlsirm-py` bindings still default
  `max_iter`, `n_starts`, `seed` (and `tol`) on `fit_bifactor_grm` and
  `fit_bifactor_grm_multigroup`, and `max_iter`/`tol` on
  `fit_bifactor_grm_fipc` (`#[pyo3(signature = (...))]` in
  `crates/fast-mlsirm-py/src/lib.rs`). A follow-up PR should remove those
  PyO3-side defaults so the two layers agree.
