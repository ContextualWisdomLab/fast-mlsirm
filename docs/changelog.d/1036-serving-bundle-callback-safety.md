# Harden serving-bundle callback boundaries

## Fixed

- Reject caller-defined serving-bundle container, schema-scalar, key, item-code, quadrature, and EAPsum-table subclasses before package validation can execute caller hashing, equality, iteration, comparison, or lookup callbacks. Valid bounded-JSON and exact built-in in-memory bundles keep the existing resource limits and Rust-owned scoring semantics.
