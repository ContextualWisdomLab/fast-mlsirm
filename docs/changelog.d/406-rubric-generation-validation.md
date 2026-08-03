# Governed rubric item generation

## Added

- Bounded source-document packets with exact-content SHA-256 provenance and redacted audit metadata.
- Content-addressed generation requests that bind one rubric contract, blueprint, seed, and evidence-mode-valid source packet.
- A runtime-checkable provider protocol and deterministic offline fixture provider without hosted SDK, credential, or network dependencies.
- Strict provider-JSON decoding that rejects duplicate keys, non-finite numbers, oversized output, excessive nesting depth, missing fields, and unknown fields.
- Immutable rubric and blueprint replay protection across ids, 128-bit audit handles, full fingerprints, and governed rubric versions.
- Exact ordered rubric-score coverage, response-format-specific typed answer keys, option/key consistency, source-id resolution, and verbatim evidence-span validation.
- Explicit pairwise left/right/tie semantics with null-only tie preferences.
- Deterministic request, candidate, and execution fingerprints plus provider-failure redaction that omits raw source and generated text.
- Public generation, candidate, answer-key, attribution, and execution APIs with complete package exports.

Structural validation remains separate from semantic review, psychometric calibration, DIF, local-dependence, exposure, drift, and governed item-bank acceptance.
