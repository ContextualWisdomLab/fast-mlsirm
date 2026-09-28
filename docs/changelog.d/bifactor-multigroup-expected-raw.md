# Multiple-group bifactor expected raw scores

## Added

- `predict_bifactor_expected_total_score(fit, theta, q_specific, *, group=None)`
  returns `E[T | theta_G]` pointwise for a fitted bifactor GRM. `theta` is any
  finite 1-D array — person EAPs with ties, in any order — so consumers no
  longer have to launder person scores through the monotonicity report's
  strictly ascending grid (unique + `return_inverse`) to read an expected raw
  score out of a fit.

## Changed

- The bifactor expected-score path now reads a
  `BifactorMultigroupFit` as well as a single-group `BifactorGrmFit`.
  Previously `check_bifactor_expected_total_score_monotonicity` rejected every
  multiple-group fit with `fit.a_general must be a non-empty 1-D array`,
  because a multiple-group fit stores `n_groups x n_items` slopes and
  `n_groups x n_items x (n_cat-1)` thresholds.
  The group axis is **selected, never flattened**: `group=<index>` names whose
  item parameters the curve uses, and `group=None` is admitted only when every
  group's `a_general`, `a_specific` and `threshold` rows are *exactly* equal —
  the identity an all-anchored fit (`anchor_mask=None`) creates by
  construction, checked rather than assumed, so no group is silently picked
  for a fit whose rows differ. A group with an estimated (non-unit)
  `specific_sd` raises instead of being approximated, because no fit object
  carries the `specific_map` that says which specific factor an item loads.
- `check_bifactor_expected_total_score_monotonicity` gained the same
  keyword-only `group` argument and now delegates its whole curve to
  `predict_bifactor_expected_total_score`, so the check and the prediction
  cannot disagree. No single-group behaviour changes.
