# Rubric generation text callback safety

## Security

- Reject caller-defined `str` subclasses at source-content, generation-contract JSON, candidate-parser JSON, static-fixture response, and live provider-output admission boundaries before caller-overridable text operations can execute.
- Preserve built-in string behavior, exact source whitespace and digests, redacted provider failures, deterministic generation provenance, and the existing Rust-owned psychometric/statistical computation boundary.
