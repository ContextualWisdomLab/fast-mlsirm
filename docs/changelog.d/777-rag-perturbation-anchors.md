# Governed RAG perturbation anchors

## Added

- Added source-free, content-addressed reference-free RAG perturbation anchors with finite preregistered construct/direction semantics for unsupported claims, contradictions, irrelevant context, required-evidence removal, citation swaps, semantic paraphrases, style-only rewrites, and unanswerable queries.
- Require distinct governed baseline and perturbed request fingerprints and fail closed on unknown perturbation semantics or malformed identities. Expected directions are validation hypotheses, not claims that an observed system actually changed or that an evaluator is ground truth.
