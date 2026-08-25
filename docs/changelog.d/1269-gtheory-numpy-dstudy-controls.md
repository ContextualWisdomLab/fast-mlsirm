# G-theory NumPy D-study control compatibility

## Fixed

- Preserve exact NumPy signed/unsigned integer arrays for one-facet and two-facet D-study size controls, and preserve exact built-in `range` values on the one-facet `Sequence[int]` surface, while continuing to reject ndarray subclasses, arbitrary array providers, callback-bearing sequence subclasses, Boolean/float/object/text control arrays, malformed rank/shape, non-positive values, and existing resource-limit violations before Rust dispatch.
- Normalize accepted NumPy control arrays and built-in range controls to package-owned built-in integer payloads; G-study, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.