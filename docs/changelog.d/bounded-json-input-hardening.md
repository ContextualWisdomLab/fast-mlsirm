# Descriptor-safe bounded JSON input for automation scripts

## Security

- Consolidated governed automation JSON readers behind a descriptor-safe shared
  loader with a 32 MiB inclusive byte bound and a non-recursive 128-level depth
  bound.
- Rejected symbolic links, FIFOs, directories, path replacement, invalid UTF-8,
  malformed JSON, non-object roots, oversized input, and excessive nesting with
  deterministic tests.
