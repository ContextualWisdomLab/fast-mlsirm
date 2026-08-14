# Harden RAG metadata callback safety

## Fixed

- Preflight caller-provided RAG metadata through the package's bounded callback-safe contract before allowlist checks, avoiding alien membership callbacks and converting hostile key iteration into non-reflective package errors.
