# Reject population-label int64 narrowing

## Security

- Reject unsigned values above the signed 64-bit boundary and floating-point
  values that would be saturated by NumPy during population-label compaction.
- Preserve the largest exact signed `int64` label while keeping group and
  cluster identifiers compact before Rust-owned allocation.
