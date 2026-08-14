# Fail-closed compiled Rust loader handling

## Fixed

- Normalize a discoverable but unloadable compiled Rust core to a package-owned runtime error while preserving the original loader exception as its cause.
