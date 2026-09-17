# Unsourced defaults removed from the polytomous DIF entry points (#1958)

## Changed

- **`dif_polytomous`, `dif_polytomous_purified`, and
  `dif_polytomous_anchor_sets` no longer default `model`, `q_theta`,
  `max_iter`, `tol`, `fdr_q`, `max_rounds` (the latter two functions), or
  `min_anchor_items`.** All are now required caller arguments. Previously
  `model` silently defaulted to `"gpcm"` (differing from the `"grm"` used
  elsewhere in this package's own study measurement models) and `q_theta`
  defaulted to `21`, a Gauss-Hermite node count with no accuracy target on
  file to source it against — the exact violation of the #1929 quadrature
  rule (node counts are caller arguments, no defaulted value below the
  project's 121-node floor) that this issue reports. `max_iter`, `tol`,
  `fdr_q`, `max_rounds`, and `min_anchor_items` have the same problem: none
  of `200`, `1e-5`, `0.05`, `3`, or `4` has a documented source in this
  repository, and this package does not ship a default it cannot defend.
- **Same reasoning already applied on this file.**
  `focal_expected_total_score_monotonicity` and
  `bifactor_expected_total_score_monotonicity` already require `q_nuisance`
  / `q_specific` with no default under #1929; this change extends that
  requirement to the sibling tuning constants on the three polytomous DIF
  functions rather than leaving them as a special case.
- **`fdr_q` and the purification loop have citable conventions, even though
  neither is defaulted.** `0.05` is the illustrative FDR level used
  throughout Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
  discovery rate: A practical and powerful approach to multiple testing.
  *Journal of the Royal Statistical Society: Series B (Methodological),
  57*(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x — and
  the anchor-rebuild-and-repeat purification loop itself is Candell, G. L.,
  & Drasgow, F. (1988). An iterative procedure for linking metrics and
  assessing item bias in item response theory. *Applied Psychological
  Measurement, 12*(3), 253-260.
  https://doi.org/10.1177/014662168801200304 — but neither source states a
  specific round count, so `max_rounds` still has no defensible default and
  stays required.
- **Breaking change, audited against the codebase's other DIF/polytomous
  entry points.** The observed-score DIF functions in `dif.py`
  (`mantel_haenszel_dif`, `mantel_haenszel_dif_purified`,
  `logistic_dif`, `logistic_dif_purified`, `sibtest`, `mantel_smd_dif`,
  `gmh_dif`, `breslow_day_dif`) and the other polytomous fit/diagnostic
  functions in `polytomous.py` were checked for the same pattern: none of
  them defaults a Gauss-Hermite node count (they either take none, or -- for
  the already-fixed `focal_expected_total_score_monotonicity` /
  `bifactor_expected_total_score_monotonicity` -- already require it), so
  they are out of scope for this issue's #1929 violation. Their `fdr_q`
  defaults are unchanged.
