# Moderated (simple) slopes on the H1–H5 OLS design

## Added

- Rust + PyO3 + Python helpers for Aiken–West / Hayes pick-a-point slopes on
  the length-10 `Y ~ X*W*Z + X*E` design: `xwz_e_design_row`,
  `design_row_dot`, `conditional_slope`, and `slope_difference`, reusing the
  existing HC sandwich `linear_contrast` path for SEs (no SciPy / Rscript).
- H1–H5 parity fixtures under `tests/data/regression_h1_h5/`
  (`beta_vcov.npz` aggregates plus coefficient/contrast CSVs) and
  `tests/test_moderated_slopes_h1_h5_parity.py` asserting estimate and HC3 SE
  to atol `1e-6` against `library_regression_h1_h5_contrasts.csv`.
