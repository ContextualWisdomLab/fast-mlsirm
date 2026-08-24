# Harden validation-policy scalar trust boundaries

## Security

- Reject caller-defined string and numeric subclasses at `ValidationPolicy` construction before `strip`, numeric conversion, or comparison callbacks can execute.
- Normalize only exact built-in and package-trusted NumPy real scalar identities for scoring-policy thresholds while preserving the existing closed `0..1` domains and Rust-owned pass/fail arithmetic.
- Require an exact built-in integer for `min_subgroup_n` before range comparison and preserve the existing `rust_kwargs()` payload contract.
