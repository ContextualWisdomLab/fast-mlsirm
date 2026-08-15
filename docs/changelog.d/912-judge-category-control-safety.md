# Judge category-count control hardening

- Validate the public `validate_judge(..., k=...)` category count before compiled-core discovery.
- Accept exact built-in integers and genuine concrete NumPy integer scalars while rejecting booleans, subclasses, and arbitrary integer-conversion protocol providers without executing caller conversion callbacks.
- Marshal only a trusted built-in integer into the existing Rust-owned judge-validation computation; psychometric/fairness formulas, thresholds, and result schemas are unchanged.

Closes #912.
