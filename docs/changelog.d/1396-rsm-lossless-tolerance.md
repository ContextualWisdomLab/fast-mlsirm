# Preserve RSM tolerance identity through Rust f64

## Fixed

- Reject Rating Scale Model `tol` controls whose exact Python or concrete NumPy integer/floating identity would change when marshalled to the Rust `f64` boundary.
- Preserve exactly representable built-in and NumPy controls, including supported `np.longdouble` values, while keeping callback-bearing scalar subclasses fail-closed before response materialization or native discovery.
- Keep RSM likelihood, marginal-ML EM/ECM updates, shared-threshold estimation, latent-trait integration, convergence, and scoring arithmetic unchanged and Rust-owned.