# Strict conformance manifest replay

## Added

- Added `ConformanceInventory.from_manifest()` and `from_json()` to rehydrate persisted cross-engine conformance evidence through exact package-owned validation.
- Persisted manifests now fail closed on unknown or missing nested keys, caller-defined mapping/list/text subtypes, duplicate JSON object keys, non-finite JSON constants, oversized JSON payloads, fingerprint tampering, and non-canonical normalized content.
- Replay remains provenance and serialization only; production psychometric and statistical arithmetic remains Rust-first.
