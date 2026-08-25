# Conformance JSON decoder depth preflight

## Fixed

- Reject conformance-manifest JSON whose structural nesting exceeds `MAX_MANIFEST_NESTING` before invoking Python's recursive JSON decoder, while preserving the exact nesting boundary and ignoring bracket/brace characters inside quoted strings and escapes.
- Preserve the existing UTF-8/byte ceiling, duplicate-member and non-finite rejection, iterative post-parse nesting validation, canonical replay, and inventory-fingerprint checks.
