# MH-RM response evidence admission

## Fixed

- Reject complex-valued MH-RM response matrices before real-valued narrowing can discard imaginary response evidence.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees containing package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Exact numeric NumPy arrays nested as inert rows inside built-in containers remain compatible without admitting ndarray subclasses or object/text leaves.
- Preserve documented `NaN` missingness, binary/GPCM category validation, and the existing Rust-owned MH-RM estimator behavior.
