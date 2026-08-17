# Hourly bounded review-repair caller

## Added

- Added a schedule-only fast-mlsirm caller that runs at minute 37 every hour and
  delegates to one immutable organization-owned review-repair workflow.
- Bounded each run to one new repair dispatch, one-hour same-head retries,
  protected `main`, non-cancelling product-level single-flight concurrency, and
  explicit scheduler credentials without direct model secrets or inherited
  secrets. A delayed next heartbeat does not discard an in-flight bounded scan;
  exact-head retry and single-writer controls remain owned by the central worker.
- Kept the workflow-generated `GITHUB_TOKEN` read-only at both workflow and call
  job scope; cross-repository mutation requires an explicitly forwarded
  established scheduler credential and fails closed when none is available.
- Added permanent caller-contract tests and APA 7th doctoring for default-branch
  activation, immutable reusable-workflow source, failure behavior, rollback,
  and the NVIDIA NIM control-plane boundary.
