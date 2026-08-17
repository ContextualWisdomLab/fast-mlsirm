# Leakage-safe model-validation units

## Added

- Added a provider-neutral `fast_mlsirm.model_validation` contract that requires model-selection validation to declare a scientific generalization unit rather than silently splitting response cells.
- Added group-partition validation that rejects one declared person/system, query/testlet, rater/family, domain/language, cluster/context, or temporal group appearing across folds.
- Added temporal-forward validation that requires an explicit temporal-period unit and rejects any window whose latest training period overlaps or follows the earliest validation period, preventing look-ahead while keeping calendar interpretation caller-owned.
- Kept predictive scoring, bootstrap statistics, likelihoods, and other result-affecting psychometric arithmetic outside this Python validation/orchestration boundary and under Rust ownership.
