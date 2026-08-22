# LLM judge score/weight inconsistency

## Fixed

- `ContextualOrchestratorJudge.judge()`'s plain scoring path (no `category_count`, the simplest and default public interface) trusted the model's own self-reported top-level `score` directly for the accept/reject decision, instead of deriving it from the per-criterion `criterion_scores` and their configured `JudgeCriterion.weight`. The `category_count`-based paths (`binary_threshold`, `cumulative_threshold`, `direct`) already discard the model's self-reported score and recompute a weight-aware average from independently validated per-criterion evidence; the plain path did not, so a criterion's weight had no effect on the outcome there, and a model could report a high aggregate score while giving a low score on the heavily-weighted criterion and still be accepted. Made the plain path derive `score` the same way as the other three, and added a regression test proving weight now changes the outcome.
