# Fail-closed compiled Rust loader handling

## Fixed

- Normalize a discoverable but unloadable compiled Rust core to a package-owned runtime error while preserving the original loader exception as its cause.
- Reject non-string and `str`-subclass backend/device control values before caller-defined conversion or normalization callbacks can execute, while preserving case/whitespace normalization for exact built-in strings.
