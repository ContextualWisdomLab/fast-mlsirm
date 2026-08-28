# Add explicit decision-support expected values

## Added

- Added a provider-neutral `fast_mlsirm.decision_support` boundary for
  caller-supplied finite state probabilities, action utilities, intervention
  costs, and coherent joint sample-information tables. Rust computes expected
  net intervention value, EVPI, and EVSI; no utility, probability, causal
  effect, or queue policy is inferred from enterprise issue text.

## Fixed

- Bound EVSI evaluation to 20,000,000 action-by-signal-by-state terms after
  shape/length admission and before probability, utility, or joint-table value
  traversal. The Rust guard uses checked cardinality arithmetic; admitted
  expected-value, EVPI, EVSI, tie, and information-cost arithmetic is unchanged.
