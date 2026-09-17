# ADR-0028 naming/defaults applied to `dif`, `deltaplot`, and `polytomous` (#1962)

## Deprecated

- **`fast_mlsirm.dif`: renamed to verb-first names.** `mantel_haenszel_dif`
  -> `detect_dif_mantel_haenszel`, `mantel_haenszel_dif_purified` ->
  `detect_dif_mantel_haenszel_purified`, `logistic_dif` ->
  `detect_dif_logistic`, `logistic_dif_purified` ->
  `detect_dif_logistic_purified`, `mantel_smd_dif` -> `detect_dif_mantel_smd`,
  `gmh_dif` -> `detect_dif_gmh`, `breslow_day_dif` -> `detect_dif_breslow_day`.
  The old names remain as deprecated aliases (identical signature and
  defaults) for one minor release, emitting `DeprecationWarning`, and stay
  exported from the same places as before (`fast_mlsirm/__init__.py` via
  `_legacy_init.py`).
- **`fast_mlsirm.polytomous`: renamed to verb-first names.**
  `bifactor_expected_total_score_monotonicity` ->
  `check_bifactor_expected_total_score_monotonicity`,
  `cat_simulate_polytomous` -> `simulate_cat_polytomous`, `dif_polytomous` ->
  `detect_dif_polytomous`, `dif_polytomous_anchor_sets` ->
  `detect_dif_anchor_sets_polytomous`, `dif_polytomous_purified` ->
  `detect_dif_polytomous_purified`, `expected_total_score_monotonicity` ->
  `check_expected_total_score_monotonicity`,
  `focal_expected_total_score_monotonicity` ->
  `check_focal_expected_total_score_monotonicity`, `information_polytomous`
  -> `compute_information_polytomous`, `item_fit_polytomous` ->
  `compute_item_fit_polytomous`, `local_dependence_polytomous` ->
  `diagnose_local_dependence_polytomous`, `person_fit_polytomous` ->
  `compute_person_fit_polytomous`, `polytomous_category_probabilities` ->
  `predict_category_probabilities_polytomous`, `polytomous_expected_response`
  -> `predict_expected_response_polytomous`, `polytomous_information_criteria`
  -> `compute_information_criteria_polytomous`, `u3_cutoff_polytomous` ->
  `compute_u3_cutoff_polytomous`, `u3_person_fit_polytomous` ->
  `compute_u3_person_fit_polytomous`. Same one-minor-release deprecated-alias
  policy as above.

## Changed

- **`fast_mlsirm.deltaplot.delta_plot`: `alpha` and `max_iter` are now
  required keyword arguments** (previously defaulted to `0.05` and `10`).
  Neither value has a documented source in this repository (ADR-0028 rule 2
  for `alpha`, a decision threshold; rule 1 for `max_iter`, an
  iteration/convergence control), so the package no longer ships a default
  it cannot defend. Every in-repo call site was updated to pass the old
  values explicitly.
- **`fast_mlsirm.dif`: `raju_area`'s `alpha` and `sibtest`'s `fdr_q`/`j_min`
  are now required keyword-only arguments** (previously `0.05`, `0.05`,
  `5`), for the same reason (ADR-0028 rule 2: no cited source for these
  specific cutoffs).
- **`fast_mlsirm.dif`: the renamed DIF entry points also drop the same
  unsourced defaults on their new names** (old aliases keep accepting the
  old default for one minor release): `detect_dif_mantel_haenszel`'s
  `fdr_q`; `detect_dif_mantel_haenszel_purified`'s `fdr_q`, `max_rounds`,
  `min_anchor_items`; `detect_dif_logistic`'s `fdr_q`, `max_iter`;
  `detect_dif_logistic_purified`'s `fdr_q`, `max_iter`, `max_rounds`,
  `min_anchor_items`; `detect_dif_breslow_day`'s `fdr_q`.
- **`fast_mlsirm.polytomous`: unsourced defaults removed from fit/scoring
  entry points.** `fit_lsirm_polytomous` (`max_iter`, `model`, `q_theta`,
  `q_xi`, `tol`), `fit_nominal_polytomous` (`max_iter`, `q_theta`, `tol`),
  `fit_poly_fipc` (`max_iter`, `q_theta`, `tol`), `fit_polytomous`
  (`max_iter`, `model`, `q_theta`, `tol`), `m2_polytomous` (`q_theta`), and
  `score_polytomous` (`q_theta`) now require these arguments explicitly —
  same #1929 quadrature-node-count rule and ADR-0028 rules 1/2/4 already
  applied to the sibling `dif_polytomous*` functions in #1958/#1960. The
  renamed functions `simulate_cat_polytomous` (`q_theta`, `se_threshold`,
  `seed`), `compute_item_fit_polytomous` (`min_expected`, `q_theta`),
  `diagnose_local_dependence_polytomous` (`q_theta`),
  `compute_person_fit_polytomous` (`flag_threshold`, `q_theta`), and
  `compute_u3_cutoff_polytomous` (`alpha`, `n_rep`, `seed`) drop the same
  category of default under their new name; old aliases keep the old
  default for one minor release.
- All in-repo call sites (package, tests, docs, examples) that relied on a
  removed default were updated to pass the old value explicitly.

## Deferred (not in this PR)

- The PyO3 entry points backing `logistic_dif`/`logistic_dif_purified`
  (`fdr_q`, `max_iter`) keep their own Rust-side defaults (`fast_mlsirm._core`
  module functions, per `docs/api/renames-and-defaults-20260917.csv`'s
  `pyo3` rows) — mirroring the Python-side default removal into the PyO3
  binding is a separate, Rust-build-required change tracked for a follow-up
  PR rather than done here (this PR is Python-only per its scope).
