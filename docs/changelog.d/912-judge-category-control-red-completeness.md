# Judge category-count RED completeness

## Security

- Extend `validate_judge(..., k=...)` regressions so `__index__`-only providers, comparison/repr hooks, booleans, `np.bool_`, 0-d arrays, and type-invalid controls fail before compiled-core discovery.
- Keep the existing trusted-scalar admission, `2..=1000` domain, and Rust-owned judge-validation arithmetic unchanged.

Closes #912.
