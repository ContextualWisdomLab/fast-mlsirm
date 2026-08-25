# Preserve Rasch CML control and group identity at the Rust boundary

## Fixed

- Reject `fit_rasch_cml()` and `andersen_lr_test()` tolerance controls that cannot be represented exactly as Rust `f64`, including wider `numpy.longdouble` values and oversized integer values, before caller response/group materialization or compiled-core discovery.
- Preserve distinct finite non-negative integral Andersen group identities carried by wider concrete NumPy floating scalars instead of narrowing them through Python `float` before deterministic dense-ID construction.
- Preserve exact built-in and supported NumPy scalar controls and group labels while keeping conditional likelihood, information, optimization, Andersen LR, p-value, and all other production psychometric arithmetic in the Rust core.
