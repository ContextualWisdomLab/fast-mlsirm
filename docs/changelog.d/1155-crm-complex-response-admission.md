# CRM response data integrity

## Fixed

- Reject complex-valued continuous-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component, and reject object-dtype response storage before caller-defined numeric conversion can run.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees with package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Text storage is rejected as non-numeric evidence. Ordinary real-valued evidence, including `NaN` missing cells, retains the existing Rust-owned CRM fitting path.
