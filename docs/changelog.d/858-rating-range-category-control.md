# Paired rating-range category control hardening

## Fixed

- Require exact built-in or package-supported genuine NumPy integer scalar identities for `paired_rating_range_evidence(..., category_count=...)`, rejecting caller-defined subclasses before conversion, type-hash/equality, representation, or Rust-dispatch callbacks can execute.
