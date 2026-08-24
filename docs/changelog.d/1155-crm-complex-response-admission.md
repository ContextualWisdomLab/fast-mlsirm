# CRM response data integrity

## Fixed

- Reject complex-valued continuous-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component, and reject object-dtype response storage before caller-defined numeric conversion can run.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees with package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Exact numeric NumPy arrays nested as inert rows inside built-in containers remain compatible without admitting ndarray subclasses or object/text leaves.
- Preserve `NaN` as the CRM missing-cell marker while rejecting `+Infinity` and `-Infinity` before native discovery instead of silently reclassifying those invalid observed values as missing. Ordinary finite real-valued evidence retains the existing Rust-owned CRM fitting path.
- Bound CRM response evidence to 20,000,000 logical cells before sequence materialization or dense real-valued work. Exact broadcast arrays and exact NumPy row leaves nested in trusted built-in matrices are rejected from shape/size metadata before allocation; shared acyclic built-in subtrees retain logical-occurrence accounting without exponential re-traversal.
