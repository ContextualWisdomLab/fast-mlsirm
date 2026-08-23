# Many-facet rating evidence admission hardening

## Fixed

- Reject complex or non-real-numeric Many-Facet Rasch response storage before real-valued marshalling so observed rating evidence cannot be silently projected onto different categories.
- Reject arbitrary top-level NumPy array providers and callback-bearing container/scalar identities before package-triggered array materialization, while preserving exact NumPy numeric arrays and ordinary exact built-in list/tuple evidence with trusted Python/NumPy real scalars.
- Bound Many-Facet Rasch response evidence to 20,000,000 logical cells before sequence materialization or dense real-valued work. Exact broadcast arrays are rejected from shape/size metadata, and built-in rating trees count trusted scalar leaves with nesting-depth-bounded traversal state before NumPy stacking.
- Preserve `NaN` missingness and existing category/domain validation for accepted real numeric arrays.
- Keep likelihood, marginal-ML EM, item difficulty, rater-severity, threshold, EAP, connectedness, and convergence arithmetic Rust-owned and unchanged.
