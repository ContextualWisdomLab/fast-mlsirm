# Harden serving-bundle callback boundaries

## Fixed

- Reject caller-defined serving-bundle container, schema-scalar, key, item-code, quadrature, and EAPsum-table subclasses before package validation can execute caller hashing, equality, iteration, comparison, or lookup callbacks. Valid bounded-JSON and exact built-in in-memory bundles keep the existing resource limits and Rust-owned scoring semantics.
- Reject serving-export factor identities that would require lossy complex/fractional/object coercion, signed narrowing, negative indices, or dimensions outside the supported `0..63` range before compiled-core discovery. Exact NumPy arrays and built-in sequences containing trusted integer-valued Python/NumPy scalars remain supported and are marshalled as contiguous `int64` identities.
