# IRT response, mask, and readiness-control admission integrity

## Fixed

- Reject complex, textual, object-backed, and arbitrary array-provider response evidence before float64 marshalling at the shared IRT response/readiness boundary.
- Preserve exact NumPy real-numeric arrays and ordinary built-in nested response containers, including supported concrete NumPy real scalar cells, `longlong`/`ulonglong` integer aliases, and NaN missingness.
- Reject callback-bearing mask evidence before Boolean coercion while preserving exact Boolean/real-numeric NumPy arrays and ordinary trusted numeric mask containers.
- Detect cyclic built-in response and mask containers with active-path identity tracking so shared acyclic rows remain valid while self/mutual cycles fail closed before NumPy materialization.
- Validate response rank, minimum persons/items, and a 20,000,000 logical-cell ceiling before contiguous float64 allocation, preventing large zero-stride or otherwise oversized exact arrays from forcing dense copies before rejection.
- Bound trusted built-in response/mask tree traversal before NumPy sequence materialization, charge logical cells hidden in exact NumPy row leaves, and reject zero-cell container fan-out that exceeds the structural-work envelope while preserving every valid 2-D matrix inside the 20,000,000-cell contract.
- Validate IRT family and category-count semantics before `fit_irt_experiment()` can materialize caller response evidence, and apply trusted response/mask admission before any production numerical fitter runs.
- Reject caller-defined integer subclasses and arbitrary integer-conversion providers at IRT experiment-readiness controls before caller callbacks can run, while preserving exact built-in and concrete NumPy integer scalar compatibility and existing readiness domains/errors.
- Keep production psychometric/statistical arithmetic Rust-owned; these changes are limited to Python validation, bounded materialization, and marshalling.
