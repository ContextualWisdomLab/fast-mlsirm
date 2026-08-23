# G-theory NumPy D-study control compatibility

## Fixed

- Preserve exact NumPy signed/unsigned integer arrays for one-facet and two-facet D-study size controls while continuing to reject ndarray subclasses, arbitrary array providers, Boolean/float/object/text control arrays, malformed rank/shape, non-positive values, and existing resource-limit violations before Rust dispatch.
- Normalize accepted NumPy control arrays to package-owned built-in integer payloads; G-study, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.
