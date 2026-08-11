# Governed RAG scoring request privacy

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

National Institute of Standards and Technology. (2020). *Security and privacy controls for information systems and organizations* (NIST SP 800-53 Rev. 5). https://doi.org/10.6028/NIST.SP.800-53r5

## Rationale

Reference-free RAG evaluation records provenance identities (system configuration, retrieval run, query revision fingerprints) without embedding raw query, context, or answer text. Managed identifiers must reject free-form content so the canonical scoring artifact cannot become a side channel.

## Implementation

`python/fast_mlsirm/scoring/rag.py` — `build_rag_scoring_request` validates `system_configuration_id` via `descriptive_identifier` and allowlists caller metadata.
