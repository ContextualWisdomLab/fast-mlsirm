# Governed reference-free RAG scoring requests

## Added

- Added a provider-neutral reference-free RAG adapter that maps query, query revision/testlet, system configuration/run, generated response, retrieval-run fingerprint, evidence regime, candidate visibility, and bounded response evidence onto the canonical `ScoringRequest` contract without carrying raw query/context/answer/source text.
- Added fail-closed package-managed provenance and a descriptive-identifier boundary for the sole caller metadata key `evaluation_split`, so raw content cannot use the allowlisted field as a canonical-artifact side channel.
- Added method doctoring grounded in RAGAS, ARES, and RAGChecker that keeps reference-free measurement construct-specific and explicitly rejects truth, absolute-recall, fairness, or deployment-validity claims from the request adapter alone.
