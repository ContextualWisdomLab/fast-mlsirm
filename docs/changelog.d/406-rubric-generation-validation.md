# Governed rubric item generation

## Added

- Bounded source-document packets with exact-content SHA-256 provenance and redacted audit metadata.
- Content-addressed generation requests that bind one rubric contract, blueprint, seed, and evidence-mode-valid source packet.
- A runtime-checkable provider protocol and deterministic offline fixture provider without adding hosted SDK, credential, or network dependencies.
- Strict provider-JSON decoding that rejects duplicate keys, non-finite numbers, oversized output, missing fields, and unknown fields.
- Exact rubric-score coverage, response-format-specific structure, source-id resolution, and verbatim evidence-span validation.
- Deterministic candidate and execution fingerprints plus provider-failure redaction that omits raw source and generated text.

This unreleased fragment will be folded into `CHANGELOG.md` after a real optional provider adapter, semantic screening, and Rust-backed calibration demonstration complete the end-to-end product path.
