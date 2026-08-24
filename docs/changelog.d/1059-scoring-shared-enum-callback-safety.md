# Scoring shared enum callback safety

## Fixed

- Shared scoring enum admission now preserves exact enum members and accepts only exact built-in strings for serialized enum values before invoking Enum lookup.
- Caller-defined string subclasses and arbitrary non-text objects fail closed with the existing package-owned assessment error before hostile hash or equality callbacks can run.
- Added public EngineDescriptor regressions proving callback-free rejection while preserving built-in string and exact enum-member compatibility; no scoring, calibration, likelihood, estimator, ranking, utility, or psychometric arithmetic changed.
