# Bifactor scoreability control callback safety

## Fixed

- Bifactor scoreability now admits `general_factor` and `zero_tolerance` through a callback-free package boundary before loading data or discovering the compiled Rust core. Caller-defined numeric subclasses, booleans, and arbitrary conversion providers are rejected while exact built-in and supported concrete NumPy scalars remain compatible. ECV, PUC, omega, construct-replicability, and all other scoreability arithmetic remain Rust-owned.
