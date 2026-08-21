# Seal governed RAG request replay

## Fixed

- Reject caller-defined `ScoringRequest` subclasses at governed RAG perturbation and facets-calibration replay boundaries before any request field can execute caller code. Exact factory-sealed requests retain the existing provenance validation, while invalid subclasses now fail through stable non-reflective package errors.
