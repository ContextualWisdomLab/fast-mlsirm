# Add explicit decision-support expected values

## Added

- Added a provider-neutral `fast_mlsirm.decision_support` boundary for
  caller-supplied finite state probabilities, action utilities, intervention
  costs, and coherent joint sample-information tables. Rust computes expected
  net intervention value, EVPI, and EVSI; no utility, probability, causal
  effect, or queue policy is inferred from enterprise issue text.
- Fail closed on impossible native result-map cardinality before result-key
  materialization, while preserving Rust ownership of all decision arithmetic.
