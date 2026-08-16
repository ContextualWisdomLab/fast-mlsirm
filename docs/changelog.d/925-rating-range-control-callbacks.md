# Harden paired rating-range category-count controls

## Changed

- Harden `paired_rating_range_evidence(..., category_count=...)` so caller-defined Python or NumPy integer subclasses cannot execute conversion callbacks before validation or native-core discovery. Exact supported NumPy integer scalars remain accepted and are normalized to a built-in `int`; Rust-owned rating-range arithmetic is unchanged.
