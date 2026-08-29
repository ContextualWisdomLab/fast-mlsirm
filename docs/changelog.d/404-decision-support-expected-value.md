# Add explicit decision-support expected values

## Added

- Added a provider-neutral `fast_mlsirm.decision_support` boundary for
  caller-supplied finite state probabilities, action utilities, intervention
  costs, and coherent joint sample-information tables. Rust computes expected
  net intervention value, EVPI, and EVSI; no utility, probability, causal
  effect, or queue policy is inferred from enterprise issue text.
- Fail closed on impossible native result-map cardinality before result-key
  materialization, while preserving Rust ownership of all decision arithmetic.
- Preflight the public 4,096-state, 1,024-action, and 1,024-signal ceilings
  from bounded inert shape metadata before boolean scans, dense binary64
  marshalling, or native dispatch; the generic 1,000,000-cell evidence ceiling
  retains precedence when both resource limits are exceeded.
- Reject an intervention-cost/action-count mismatch from bounded shape metadata
  after the existing dimensionality, non-empty, and generic cell guards but
  before Boolean/value scans or contiguous binary64 cost marshalling.
