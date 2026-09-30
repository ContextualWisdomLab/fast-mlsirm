# Multigroup EM monotone population update (#2093)

## Fixed

- `fit_bifactor_grm_multigroup` no longer fails with "EM observed-data
  log-likelihood decreased" on small or near-separable data. The focal-group
  `mu`/`sigma` (and `tau` when `estimate_specific_vars=True`) update is now an
  ECM conditional-maximization step (Meng & Rubin, 1993, p. 269) on the same
  expected complete-data log-likelihood as the item step, so every EM iteration
  is a GEM step and the guarded quadrature log-likelihood never decreases
  (Dempster, Laird, & Rubin, 1977, Theorem 1, p. 7). The posterior-moment
  update it replaces moved the quadrature nodes without ascending that
  objective. The MML target is unchanged; at a finite node count the fitted
  focal distribution moves to the stationary point of the reported quadrature
  log-likelihood (differences are of quadrature-error order and vanish as the
  node count grows). The fixed-item calibration path is unchanged.
