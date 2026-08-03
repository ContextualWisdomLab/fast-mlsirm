# Rubric blueprint compiler

## Added

- Versioned rubric and rubric-level schemas with explicit construct, observable evidence, task-family, response-format, locale, and prohibited-pattern contracts.
- Deterministic bounded compilation across task family, difficulty band, evidence mode, and replicate cells.
- SHA-256 content-addressed rubric fingerprints, item-blueprint ids, generation seeds, and provider-neutral structured-output contracts.
- A prompt-injection boundary and strict generated-item JSON Schema 2020-12 contract without adding a hosted-model SDK or network dependency.
- Ordered score-level schemas that require every rubric score exactly once in scoring guides and alignment evidence, preventing duplicate-level output from omitting a score category.
- Explicit text and collection bounds for model-generated content and provenance fields.
- Evidence-Centered Design documentation and a production roadmap from provider adapters through Rust-backed calibration and governed item-bank lifecycle.

This unreleased fragment will be folded into `CHANGELOG.md` when the end-to-end provider, screening, and calibration vertical path is release-ready.
