# Harden configuration integer trust boundaries

## Fixed

- Reject caller-defined integer subclasses and arbitrary `__index__` providers before public simulation and fit configuration validation can dispatch caller-controlled coercion.
- Preserve exact built-in integers and genuine NumPy integer scalars while validating simulation size, optimizer-work, quadrature, and latent-integration controls through built-in integer values.
- Run the same simulation and fit validators at construction so memory-safety bounds cannot be bypassed by skipping an explicit `validate()` call.
