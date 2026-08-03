# Rubric blueprint compiler

## Added

- Versioned rubric and rubric-level schemas with explicit construct, observable evidence, task-family, response-format, locale, and prohibited-pattern contracts.
- Deterministic bounded compilation across task family, difficulty band, evidence mode, and replicate cells.
- Full SHA-256 rubric, blueprint, and generation-contract fingerprints plus authoritative 128-bit public blueprint and contract handles; 64-bit convenience display identifiers remain explicitly non-authoritative, and 64-bit digest slices also seed deterministic generation.
- A prompt-injection boundary and strict generated-item JSON Schema 2020-12 contract without adding a hosted-model SDK or network dependency.
- Immutable rubric and blueprint provenance constants in generated-item schemas, preventing wrong-blueprint replay from passing structural validation.
- Response-format-specific, closed, bounded answer-key contracts and ordered score-level schemas that require every rubric score exactly once.
- Explicit text and collection bounds for model-generated content and provenance fields.
- A deterministic standard-library changelog-fragment renderer; files in `docs/changelog.d` are authoritative `Unreleased` release notes and are validated as part of the repository test suite.
- Evidence-Centered Design documentation and a production roadmap from provider adapters through Rust-backed calibration and governed item-bank lifecycle.
