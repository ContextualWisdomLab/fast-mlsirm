# Harden Rudner/Lee cut-score control admission

## Fixed

- Validate and materialize Rudner and Lee cut-score scalars before compiled Rust capability discovery, rejecting booleans, caller-defined scalar subclasses, protocol coercion providers, malformed containers, non-finite values, and conversion overflow without invoking caller conversion hooks while preserving exact built-in and concrete NumPy real scalar compatibility. Both public paths now use one canonical package-owned normalizer; cut ordering/domain checks and all classification arithmetic remain Rust-owned.
