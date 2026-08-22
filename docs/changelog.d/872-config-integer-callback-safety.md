# Harden configuration integer callback boundaries

## Fixed

- Reject arbitrary integer protocols and caller-defined integer subclasses in MLS2PLM simulation-size controls plus `FitConfig.lbfgs_history`, `xi_points`, and `xi_seed` before caller callbacks or comparisons can run, while preserving exact built-in and supported concrete NumPy integer scalars and existing resource/domain limits.
