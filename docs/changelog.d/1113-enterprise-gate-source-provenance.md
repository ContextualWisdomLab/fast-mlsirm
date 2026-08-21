# Enterprise gate source-provenance hardening

## Fixed

- Require enterprise due-diligence manifests to bind `source_commit` to a canonical lowercase full SHA-1 or SHA-256 Git object identity instead of accepting abbreviated or arbitrary printable identifiers.
- Reject caller-defined string subclasses before text callbacks can execute at the source-provenance admission boundary, so a successful gate remains reconstructable from exact source identity.
- Restrict manifest output to a relative path inside the invocation directory and reject symlinked or tree-escaping destinations before writing.
- Write through a validated descriptor tree into a same-directory temporary file and atomically rename it into place on supported POSIX systems, so a failed write cannot truncate the previously accepted manifest.
- Preserve an existing manifest's access permissions across atomic replacement and use ordinary process file-creation permissions for a new manifest instead of forcing buyer-facing evidence to owner-only mode.
