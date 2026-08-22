# Mokken input admission

## Fixed

- Validate Mokken AISP scalar controls and score storage before compiled-core discovery, reject complex/object response evidence before numeric narrowing, and reject unsigned or floating category values outside signed `int64` before Rust marshalling.
- Preserve the historical scalar semantics of exact zero-dimensional numeric NumPy arrays for `lower_bound` and `alpha` while continuing to reject ndarray subclasses, object/complex storage, booleans, and arbitrary caller conversion protocols.
- Loevinger scalability, Z-statistics, and AISP arithmetic remain unchanged and Rust-owned.
