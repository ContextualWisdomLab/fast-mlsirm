# Harden configuration integer trust boundaries

## Fixed

- Reject caller-defined integer subclasses and arbitrary `__index__` providers before public simulation and fit configuration validation can dispatch caller-controlled coercion.
- Preserve exact built-in integers and genuine NumPy integer scalars while validating simulation size, optimizer-work, quadrature, latent-integration, seed, and verbosity controls through built-in integer values.
- Store those trusted integers back on the frozen configs so later size products and `seed + restart` cannot wrap narrow NumPy scalars.
- Normalize `dimensionality_diagnostics` `k_folds`, `seed`, and `latent_dims` to built-in integers before the candidate-by-fold budget product or `seed + fold_idx` can wrap a narrow NumPy scalar.
- Run the same simulation and fit validators at construction so memory-safety bounds cannot be bypassed by skipping an explicit `validate()` call.
