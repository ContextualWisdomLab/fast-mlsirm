# Enterprise gate source-provenance hardening

## Fixed

- Require enterprise due-diligence manifests to bind `source_commit` to a canonical lowercase full SHA-1 or SHA-256 Git object identity instead of accepting abbreviated or arbitrary printable identifiers.
- Reject caller-defined string subclasses before text callbacks can execute at the source-provenance admission boundary, so a successful gate remains reconstructable from exact source identity.
