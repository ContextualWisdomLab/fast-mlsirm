### Fixed

- Hardened `ValidationPolicy` scalar admission so caller-defined string, float, and integer subclasses cannot execute coercion, comparison, or text callbacks before Rust-owned validation decisions; trusted built-in values and the legacy exact NumPy `float64` threshold scalar remain supported.
