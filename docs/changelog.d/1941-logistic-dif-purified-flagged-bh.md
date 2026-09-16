# `logistic_dif_purified` purifies again, on `flagged_bh` (#1941)

## Fixed

- **`logistic_dif_purified` no longer no-ops.** Its anchor-purification
  criterion was `purify_flagged(jg_class)` (`jg_class in {B, C}`). #1880
  retired `jg_class` to `"U"` ("not applicable") for every item, so that
  criterion could never fire: `n_anchor` stayed at the initial item count and
  `rounds` stayed `0` regardless of the DIF actually present in the data.
  This silently downgraded the function to an expensive wrapper around
  `logistic_dif` that never purified anything.
- **Replacement criterion: `flagged_bh`.** The purification loop now drops an
  item from the anchor when its Benjamini-Hochberg-adjusted `chi2_total`
  omnibus test (`flagged_bh`) rejects — the same multiplicity-controlled
  significance test `dif_polytomous_purified` uses for the identical reason:
  no calibrated practical-significance class (an ETS-style B/C letter)
  exists for this statistic. Jodoin and Gierl's (2001) `.035`/`.070` bands
  are stated on a Zumbo-Thomas weighted-least-squares one-degree-of-freedom
  partition this package does not compute, not on the two-degree-of-freedom
  Nagelkerke pseudo-R² `delta_r2` this package reports (see #1880's
  fragment), so `flagged_bh` is used directly rather than guessing a class.
  This makes the loop's anchor MORE aggressive at large `N` than
  `mantel_haenszel_dif_purified`'s practical-significance screen, not less —
  documented on the function.
- **No public API change.** `logistic_dif_purified`'s signature and return
  keys (`anchor`, `n_anchor`, `rounds`, `purify_converged`,
  `purify_termination_reason`, plus every `logistic_dif` key) are unchanged.
  Only the purification behavior — which items the anchor excludes, for data
  with DIF present — changes, from "never" to "on `flagged_bh`".
  `jg_class` itself is untouched and remains `"U"` for every item.
- **Caveat inherited, not introduced.** As before, the anchor is selected
  from the same data it is then tested against, so the returned p-values are
  conditional on a data-dependent selection and Benjamini-Hochberg does not
  carry an FDR guarantee for the purified sweep; treat `flagged_bh` as a
  screening device (see the function's existing docstring caveats, unchanged
  by this fix).
- **References.** Candell, G. L., & Drasgow, F. (1988). An iterative
  procedure for linking metrics and assessing item bias in item response
  theory. *Applied Psychological Measurement, 12*(3), 253-260.
  https://doi.org/10.1177/014662168801200304 — Zumbo, B. D. (1999). *A
  handbook on the theory and methods of differential item functioning
  (DIF): Logistic regression modeling as a unitary framework for binary and
  Likert-type (ordinal) item scores* (p. 27). Directorate of Human Resources
  Research and Evaluation, Department of National Defense.
