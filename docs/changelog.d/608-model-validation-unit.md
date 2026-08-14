# Leakage-safe model-validation units

## Added

- Added a provider-neutral `fast_mlsirm.model_validation` contract that requires model-selection validation to declare a scientific generalization unit rather than silently splitting response cells.
- Added group-partition validation that rejects one declared person/system, query/testlet, rater/family, domain/language, cluster/context, or temporal group appearing across folds, plus a temporal-forward contract that requires explicit temporal-period validation.
- Kept predictive scoring, bootstrap statistics, likelihoods, and other result-affecting psychometric arithmetic outside this Python validation/orchestration boundary and under Rust ownership.
