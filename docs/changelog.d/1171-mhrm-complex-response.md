# MH-RM response and control admission

## Fixed

- Reject complex-valued MH-RM response matrices before real-valued narrowing can discard imaginary response evidence.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees containing package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Exact numeric NumPy arrays nested as inert rows inside built-in containers remain compatible without admitting ndarray subclasses or object/text leaves.
- Admit MH-RM family, iteration, proposal/tolerance, seed, and uncertainty/correlation controls before caller-owned response work or compiled-core discovery; normalize supported concrete Python/NumPy scalars to built-in Rust-boundary primitives and reject callback-bearing identities without executing their conversion protocols.
- Mirror the Rust-owned unsigned iteration domains before response materialization, including negative/zero cycle and Metropolis-step values that cannot enter the native `usize` configuration.
- Preserve documented `NaN` missingness, binary/GPCM category validation, and the existing Rust-owned MH-RM stochastic estimation, latent-correlation, uncertainty, convergence, and recovery arithmetic.