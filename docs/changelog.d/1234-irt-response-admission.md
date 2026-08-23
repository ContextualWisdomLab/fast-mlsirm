# IRT response, mask, and readiness-control admission integrity

## Fixed

- Reject complex, textual, object-backed, and arbitrary array-provider response evidence before float64 marshalling at the shared IRT response/readiness boundary.
- Preserve exact NumPy real-numeric arrays and ordinary built-in nested response containers, including supported concrete NumPy real scalar cells and NaN missingness.
- Reject callback-bearing mask evidence before Boolean coercion while preserving exact Boolean/real-numeric NumPy arrays and ordinary trusted numeric mask containers.
- Validate IRT family and category-count semantics before `fit_irt_experiment()` can materialize caller response evidence, and apply trusted response/mask admission before any production numerical fitter runs.
- Reject caller-defined integer subclasses and arbitrary integer-conversion providers at IRT experiment-readiness controls before caller callbacks can run, while preserving exact built-in and concrete NumPy integer scalar compatibility and existing readiness domains/errors.
- Keep production psychometric/statistical arithmetic Rust-owned; these changes are limited to Python validation, bounded materialization, and marshalling.
