# Mokken input admission

## Fixed

- Validate Mokken AISP scalar controls and score storage before compiled-core discovery, reject complex/object response evidence before numeric narrowing, and reject unsigned or floating category values outside signed `int64` before Rust marshalling.
- Reject caller-defined response array providers, container subclasses, and numeric subclasses before NumPy protocol execution while preserving exact NumPy arrays, ordinary built-in rows, and exact NumPy row arrays composed of supported real numeric evidence.
- Preserve the historical scalar semantics of exact zero-dimensional numeric NumPy arrays for `lower_bound` and `alpha` while continuing to reject ndarray subclasses, object/complex storage, booleans, and arbitrary caller conversion protocols.
- Keep unsigned signed-`int64` overflow detection exact across the supported NumPy 1.x/2.x range by comparing against an unsigned NumPy boundary instead of relying on value-based Python-int promotion.
- Loevinger scalability, Z-statistics, and AISP arithmetic remain unchanged and Rust-owned.
