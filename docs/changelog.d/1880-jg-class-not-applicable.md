# `logistic_dif`'s `jg_class` is retired to "not applicable" (#1880)

## Changed

- **Breaking, for any caller reading `jg_class`.** `logistic_dif` (and
  `logistic_dif_purified`) no longer letter items `"A"`/`"B"`/`"C"` against the
  Jodoin and Gierl (2001) effect-size bands. `jg_class` now reports `"U"`
  ("not applicable") unconditionally, for every item, regardless of fit
  success or omnibus significance. `delta_r2` (the Nagelkerke `R2(M2) -
  R2(M0)` change across the 2-df omnibus) and `delta_r2_uniform` (`R2(M1) -
  R2(M0)`) are unchanged and remain reported as descriptive numbers; neither
  carries a letter class. Callers who branched on `jg_class == "A"`/`"B"`/`"C"`
  must switch to reading `delta_r2` / `delta_r2_uniform` directly, or to
  `flagged_bh` for a significance-only decision.
- `logistic_dif_purified`'s anchor no longer shrinks: its purification
  criterion (`purify_flagged`) only fires on `jg_class in {B, C}`, which is
  now unreachable. This is deliberate, not a silent regression: the obvious
  substitute, raw `flagged_bh` significance, is exactly the over-powered
  test this package's purification design exists to screen against (see the
  doc comment on `logistic_dif_purified`), so substituting it would purify
  far more aggressively than the retired letter class ever did — a
  materially different and worse behaviour change than simply not
  purifying. Callers who relied on `logistic_dif_purified` actually
  shrinking the anchor should track #1880 for a principled replacement
  criterion, or use `mantel_haenszel_dif_purified` (unaffected; it purifies
  on its own ETS `ets_class`, not `jg_class`).

- **Why.** Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I error and
  power rates using an effect size measure with the logistic regression
  procedure for DIF detection. *Applied Measurement in Education, 14*(4),
  329-349, calibrates its `.035`/`.070` bands (p. 335) on a Zumbo-Thomas
  weighted-least-squares (Pratt-Pregibon) partition (p. 333) — not the
  Nagelkerke pseudo-R² this package computes — and states them on the
  ONE-degree-of-freedom UNIFORM increment, not the 2-df omnibus this package
  previously lettered (`delta_r2`). Both mismatches were confirmed against
  the primary source (read in full via institutional access) and would need
  correcting together, but the replacement statistic is itself
  underdetermined by that source: eq. 4 (p. 333) does not say whether its
  correlation is taken against the observed response or the working response
  of the IRLS linearization, nor on which scale the standardized coefficient
  is computed, and both choices change the number. A package cannot letter a
  quantity it cannot compute, so the honest fix is "not applicable," not a
  guessed replacement. The bands are additionally scoped to the paper's own
  simulation — dichotomous responses generated from a 3PL model (pp. 337,
  339) on 40-item tests (pp. 336-337) — so they were never licensed for the
  polytomous logistic-regression sweep either (see PR #1890's independent
  finding on this point).
- **Migration.** Any stored or logged `jg_class` values from before this
  change reflected bands applied to the wrong quantity (2-df omnibus instead
  of 1-df uniform) computed from the wrong statistic (Nagelkerke instead of
  the Zumbo-Thomas WLS partition); they should not be treated as ground truth
  and do not need to be "corrected" to a new letter, because no letter is
  defensible. Use `delta_r2` / `delta_r2_uniform` (continuous, unchanged) and
  `flagged_bh` (significance only, unchanged) directly.
- **References.** Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I
  error and power rates using an effect size measure with the logistic
  regression procedure for DIF detection. *Applied Measurement in Education,
  14*(4), 329-349. https://doi.org/10.1207/S15324818AME1404_2 — Nagelkerke,
  N. J. D. (1991). A note on a general definition of the coefficient of
  determination. *Biometrika, 78*(3), 691-692.
  https://doi.org/10.1093/biomet/78.3.691 — Zumbo, B. D. (1999). *A handbook
  on the theory and methods of differential item functioning (DIF)*.
  Directorate of Human Resources Research and Evaluation, Department of
  National Defense.
