# Preserve Rasch CML tolerance identity at the Rust boundary

## Fixed

- Reject `fit_rasch_cml()` and `andersen_lr_test()` tolerance controls that cannot be represented exactly as Rust `f64`, including wider `numpy.longdouble` values and oversized integer values, before caller response/group materialization or compiled-core discovery.
- Preserve exact built-in and supported NumPy scalar tolerances while keeping conditional likelihood, information, optimization, Andersen LR, p-value, and all other production psychometric arithmetic in the Rust core.
