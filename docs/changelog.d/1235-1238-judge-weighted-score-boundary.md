# Bound the judge's weighted-score boundary

## Fixed

- `ContextualOrchestratorJudge.judge()`'s plain scoring path (no `category_count`, the simplest public interface) trusted the model's own self-reported top-level `score` for the accept/reject decision instead of deriving it from `criterion_scores` and each `JudgeCriterion.weight`, unlike the three `category_count`-based paths, which already discard the self-reported score in favor of a mechanically recomputed weight-aware average. A model could report a high aggregate score while giving a low score on a heavily-weighted criterion and still be accepted. Made the plain path derive `score` the same way as the other three (issue #1238).
- Rejected a non-finite aggregate criterion weight before any contextual-orchestrator transport call. `JudgeCriterion` validates each weight as finite and positive, but two individually valid weights (for example `1e308` each) could still overflow their sum to infinity; a weighted score could then silently collapse to an incorrect finite value (for example `0.0`) instead of failing closed. All three weighted-score paths now share one bounded, finite denominator (issue #1235).
