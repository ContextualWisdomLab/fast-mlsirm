# Governed RAG perturbation anchors

## Added

- Added source-free, content-addressed reference-free RAG perturbation anchors with finite preregistered construct/direction semantics for unsupported claims, contradictions, irrelevant context, required-evidence removal, citation swaps, semantic paraphrases, style-only rewrites, and unanswerable queries.
- Require canonical governed baseline and perturbed `ScoringRequest` values, reject unrelated or mixed-axis pairs, and bind each anchor to exact perturbation specification/run fingerprints while serializing only source-free identities.
- Distinguish literature-aligned constructs from package-owned model-design hypotheses. Every expected direction remains a validation hypothesis, not a claim that the cited papers established the exact perturbation, that an observed system actually changed, or that an evaluator is ground truth.
