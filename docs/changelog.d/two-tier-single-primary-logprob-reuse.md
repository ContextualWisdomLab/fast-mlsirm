# Two-tier GRM single-primary fit speed: September 20, 2026 record

## Fixed

- The September 20, 2026 PR #2070 implementation addressed a single-primary
  two-tier graded response slowdown in 0.11.4. That implementation filled
  category log-probabilities once per E-step sweep and final EAP pass, then
  reused them across persons. On the recorded local 500-person, 6-item,
  15 × 11 quadrature fixture, warm median wall times were 1.75 s for 0.11.3,
  16.19 s for 0.11.4, and 1.61 s for the historical fix. The host load was
  elevated; these are retained measurements, not a current speedup claim.
- The historical implementation retained memory-bounded streamed evaluation
  for multi-primary fits. It did not change quadrature counts, tolerances,
  iteration caps, or the model formula. This describes that historical
  implementation, not the current implementation.

## Changed

- Consumer commit `96b3397e7890dd062c0bffd8c8f4e58298792005` uses the ordinary
  exact-primary-predictor probability bank across primary dimensions; it does
  not contain the historical `fill_logprob_tables` helper. No historical
  core code is reintroduced with this record.
- `billing-snapshots/perf_0114_regression_20260920.json` preserves the original
  dated workload, commands, source/binary identities, raw-log hashes, and
  observations. Its native binaries and scientific outcomes require their
  own historical provenance checks; importing the record does not attest a
  currently loaded binary or a current completed fit.
- `tests/test_perf_0114_reproduction_contract.py` tests the retained shell
  commands with synthetic Cargo/timer adapters. It checks working-directory
  and repetition-failure propagation, not compilation or fitting.
- The historical 15 × 11 fixture does not satisfy the current caller-selected
  121-or-more quadrature refinement, full-reference convergence, precision,
  negative recovery, full-step, absolute-log-likelihood, no-rollback, or
  two-host GPU acceptance requirements. Those requirements remain unchanged.
