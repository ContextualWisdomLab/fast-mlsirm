### Fixed

- Hardened `ValidationPolicy` semantic controls so caller-defined string, float, and integer subclasses cannot execute coercion or comparison callbacks before validation decisions reach the Rust agreement kernel. Existing built-in policy values and Rust-owned threshold arithmetic remain unchanged.
