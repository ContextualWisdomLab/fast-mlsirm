# Interaction-map input digest provenance

## Added

- Bind each versioned residual interaction-map envelope to a deterministic SHA-256 identity of the already validated schema, requested axis count, ordered opaque person/item identifiers, matrix shape, and canonical little-endian float64 observed/expected evidence. Rust independently derives the same canonical digest, rejects a supplied mismatch before numerical map work, persists the canonical identity, and returns it for public Python replay before numerical payload marshalling.
- Seal every NumPy index/evidence array returned by the public envelope as read-only so caller mutation cannot leave changed persisted evidence attached to the original input digest.
