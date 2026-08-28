# Add Rust-owned ordered proficiency posterior summaries

## Added

- Add a framework-neutral Rust API that converts already-calibrated posterior draws and immutable ordered cut scores into normalized level probabilities, a weighted posterior mean and standard deviation, a shortest contiguous credible-level set, and an ambiguity-preserving modal-level decision.
- Preserve ordinal semantics: exact cut scores enter the upper level, tied modal probabilities do not force a reported level, and cut scores are never reordered or repaired.
- Rank equally short credible intervals by retained posterior mass, then use the lower start only when mass also ties; suppress a unique modal reported level when that modal level lies outside the selected credible interval.
- Add fail-closed validation for empty or non-finite draws, malformed weights, non-increasing cut scores, invalid credible mass, and non-finite posterior moments, plus joint draw/weight permutation regressions.
- This bounded numerical slice does not estimate ability, validate standard setting or CEFR linking, average ordinal labels, implement CAT, or authorize certification decisions.
- Research basis: Kang, I., & Jeon, M. (2025). [Multidimensional Latent Space Item Response Models: A Note on the Relativity of Conditional Dependence](https://doi.org/10.1017/psy.2025.5), *Psychometrika, 90*(2), 799–826. Their posterior-chain workflow motivates distinguishing posterior-draw spread from a repeated-sampling estimator standard error; this API reports the former and does not claim the latter.
- Research basis: Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (1998). [The Generalized Graded Unfolding Model](https://www.ets.org/research/policy_research_reports/publications/report/1998/hxxo.html), ETS Research Report RR-98-32. Its ordered-category threshold treatment supports retaining supplied cut-score order and explicit boundary semantics; this API consumes calibrated cut scores and does not estimate thresholds.
