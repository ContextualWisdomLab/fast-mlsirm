# Require interoperable bounded JSON artifacts

## Fixed

- Reject duplicate object member names and non-standard non-finite numeric constants in the shared repository-automation bounded JSON reader, so file-backed and direct parsing use the same unambiguous RFC-compatible semantics while preserving existing size, depth, UTF-8, path-identity, and callback-safety controls.
