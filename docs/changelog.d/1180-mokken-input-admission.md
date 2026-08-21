# Mokken input admission

## Fixed

- Validate Mokken AISP scalar controls and score storage before compiled-core discovery, reject complex/object response evidence before numeric narrowing, and reject unsigned or floating category values outside signed `int64` before Rust marshalling. Loevinger scalability, Z-statistics, and AISP arithmetic remain unchanged and Rust-owned.
