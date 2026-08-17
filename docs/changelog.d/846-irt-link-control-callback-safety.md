# Fail-closed IRT linking control values

## Changed

- Validate `irt_link(method=...)` as an exact built-in string against the existing Rust `LinkMethod` vocabulary before loading or calling the native core.
- Validate `q_theta` as either an exact built-in Python integer or a genuine NumPy integer scalar before native-loader access; integer subclasses are rejected before caller-controlled `__int__`/representation callbacks can run.
- Reject hostile method objects, string subclasses, unsupported method identities, and hostile quadrature subclasses with package-owned `ValueError` evidence, while preserving trusted Rust-supported aliases and genuine NumPy integer quadrature scalars.
- Keep all IRT scale-linking coefficients, characteristic-curve criteria, optimization, convergence arithmetic, and quadrature generation behavior in their existing numerical owners; this change is limited to Python validation and marshalling.
