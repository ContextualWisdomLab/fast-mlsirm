# Fail-closed IRT linking method controls

## Changed

- Validate `irt_link(method=...)` as an exact built-in string against the existing Rust `LinkMethod` vocabulary before loading or calling the native core.
- Reject hostile objects, string subclasses, and unsupported method identities without executing caller-controlled representation or normalization callbacks, while preserving trusted Rust-supported aliases unchanged.
- Keep all IRT scale-linking coefficients, characteristic-curve criteria, optimization, and convergence arithmetic in Rust; this change is limited to Python validation and marshalling.
