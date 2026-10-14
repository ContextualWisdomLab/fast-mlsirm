# Multigroup EM monotone population update (#2093)

## Fixed

- `fit_bifactor_grm_multigroup` no longer fails with "EM observed-data
  log-likelihood decreased" on small or near-separable data. The focal-group
  `mu`/`sigma` (and `tau` when `estimate_specific_vars=True`) update is now a
  bounded GEM population-ascent step on the same expected complete-data
  log-likelihood as the item step. It accepts only Armijo-improving updates and
  may stop after the fixed Newton depth before reaching the conditional maximum,
  so it is not claimed to be an ECM CM-step. Within the guarded roundoff
  tolerance, the quadrature log-likelihood does not decrease (Dempster, Laird,
  & Rubin, 1977, Theorem 1, p. 7). The posterior-moment update it replaces moved
  the quadrature nodes without ascending that objective. The MML target is
  unchanged. The fixed-item calibration path is unchanged.
