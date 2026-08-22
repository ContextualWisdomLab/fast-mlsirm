# LLM judge weighted-score trust boundary

## Fixed

- Reject aggregate criterion weights that overflow or become non-finite before any contextual-orchestrator transport call.
- Derive the authoritative judge score and accept/reject decision from validated weighted criterion evidence while retaining validation of the model-reported aggregate score field.
- Reuse one finite package-owned weight denominator across direct and category-based judge paths; LLM outputs remain fallible rater evidence rather than truth.
