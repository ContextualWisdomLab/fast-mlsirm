# Share one judge-result IRT projection core

## Changed

- Remove the duplicate category/binning implementation from the explicit criterion-order adapter. `LLMJudgeResult.to_irt_row` remains the single package-owned projection authority; the explicit-order adapter preserves its stricter sealed-mapping checks, delegates once to that canonical projection, and only permutes the validated row into caller-supplied criterion order.
- Add parity regressions for score-derived and explicit-category projections, custom criterion order, delegation to the canonical core, and the sealed result-mapping boundary. No judge scoring threshold, category arithmetic, psychometric likelihood, or Rust numerical behavior changes.
