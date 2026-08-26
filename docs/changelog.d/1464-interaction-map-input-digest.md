# Interaction-map input digest provenance

## Added

- Bind each versioned residual interaction-map envelope to a deterministic SHA-256 identity of the already validated schema, requested axis count, ordered opaque person/item identifiers, matrix shape, and canonical little-endian float64 observed/expected evidence. Rust validates and returns the digest before numerical work, and the public Python binding rejects a returned digest mismatch before numerical payload marshalling.