# Bound CDM evidence before dense materialization

## Fixed

- Reject response and Q-matrix evidence above 20,000,000 logical cells during callback-free preflight, including oversized exact NumPy leaves nested in trusted built-in sequences, before NumPy materialization or `float64` allocation while preserving existing valid evidence and Rust-owned CDM arithmetic.
- Memoize trusted shared-sequence subtree sizes so repeated acyclic DAGs retain per-occurrence logical-cell accounting without exponential re-traversal, while true cycles still fail closed.
- Keep Boolean response round-trip validation compatible with the declared NumPy floor by reserving `equal_nan=True` for floating response arrays; non-floating admitted evidence uses ordinary exact equality.
