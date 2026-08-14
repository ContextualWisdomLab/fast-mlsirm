# Harden RAG metadata callback safety

## Fixed

- Validate caller-provided RAG metadata keys exactly once before reading any values, then freeze only the captured allowlisted values. Hostile membership, key/value, duplicate-key, and key-reiteration callbacks now fail through non-reflective package errors without granting new metadata authority.
