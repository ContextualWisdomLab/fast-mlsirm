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
- Reject exact built-in state/action/signal cardinality and intervention-cost
  count contradictions from package-owned outer-container metadata before
  scalar element validation when the generic cell ceiling cannot take
  precedence; arbitrary protocols remain rejected and Rust retains all
  expected-value arithmetic.
- Reject an intervention-cost/action-count mismatch from bounded shape metadata
  after the existing dimensionality, non-empty, and generic cell guards but
  before Boolean/value scans or contiguous binary64 cost marshalling.
- Reject action-utility and sample-information state-axis mismatches from bounded
  shape metadata after the target carrier's existing resource guards but before
  Boolean/value scans or contiguous binary64 marshalling.
- Admit independently decidable `no_action_index` and `information_cost`
  controls before traversing scientific evidence, while retaining the
  action-index upper-bound check after the action count is known.
