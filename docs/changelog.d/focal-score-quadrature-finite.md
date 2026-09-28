# Finite focal expected-score quadrature at high node counts

## Fixed

- The focal-factor expected-total monotonicity check accepts `q_nuisance` up
  to 4096, but NumPy `hermegauss` weights become non-finite from 481 nodes, so
  the curve could carry non-finite values. It now uses the same Golub-Welsch
  Gauss-Hermite rule as the two-tier path.
- The nuisance scale is computed with `np.hypot.reduce` instead of
  `sum(square) - square(focal)`, which cancelled to zero for a dominant focal
  loading.
