# Opt-in EM progress for long two-tier / bifactor GRM fits (#2021)

## Added

- Add optional `progress` callable on `fit_two_tier_grm` and `fit_bifactor_grm`
  (default `None`, silent — 0.11.4-compatible). Each E-step reports
  `EmIterationProgress(iteration, loglik, delta_loglik, start)` using the
  already-computed observed-data marginal log-likelihood (Bock & Aitkin,
  1981, *Psychometrika, 46*(4), pp. 445, 447–448). No extra quadrature.
- Rust companions `fit_*_with_progress` keep existing silent entry points
  unchanged; PyO3 detaches only when `progress is None`.
