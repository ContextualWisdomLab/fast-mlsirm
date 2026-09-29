# Polytomous monotonicity checks: finite Gauss-Hermite rule at high q

## Fixed

- `check_focal_expected_total_score_monotonicity` and
  `check_bifactor_expected_total_score_monotonicity` built their nuisance /
  specific-factor rule with `numpy.polynomial.hermite_e.hermegauss`, whose
  closed-form weights underflow to all-zero from `q = 371` and overflow to NaN
  from `q = 400` (numpy 2.5.2). The expected-total curve was then all NaN and
  the report still said `monotone=True`. Both now use one shared
  Golub-Welsch rule (`polytomous._probabilists_gauss_hermite`, the algorithm of
  the Rust core's `gauss_hermite_probabilists`), finite for every `q >= 1`.
- `q_nuisance` / `q_specific` no longer carry the `MAX_POLY_QUADRATURE_POINTS`
  (4096) cap: any exact integer `>= 1` is accepted, and an unrepresentable
  `q x q` Jacobi matrix or an allocation / eigensolver failure raises
  `ValueError`.
- The shared expected-score monotonicity reducer now fails closed: a
  non-finite expected-total curve raises `ValueError` instead of reducing to
  `monotone=True` (NaN compares false, so no decrease was ever recorded).
