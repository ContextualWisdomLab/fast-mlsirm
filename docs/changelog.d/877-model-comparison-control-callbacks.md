# Model-comparison callback-boundary hardening

## Security

- Harden parameter-count, audit-label, and real-valued model-comparison controls so caller-defined integer/string/NumPy subclasses and arbitrary integer-protocol providers are rejected before conversion or normalization callbacks execute, while preserving genuine NumPy scalar compatibility and Rust-owned Vuong arithmetic.
