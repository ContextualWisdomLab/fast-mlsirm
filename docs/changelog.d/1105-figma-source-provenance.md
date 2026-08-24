# Figma evidence source provenance

## Fixed

- Figma design-evidence manifests now fail closed when the repository source commit cannot be resolved to a canonical full lowercase SHA-1 or SHA-256 object identity, instead of emitting buyer-facing evidence with `source_commit: "unknown"` or an abbreviated/malformed revision.
