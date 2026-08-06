# Hourly bounded review-repair caller

## Added

- Added a schedule-only fast-mlsirm caller that runs at minute 37 every hour and
  delegates to one immutable organization-owned review-repair workflow.
- Bounded each run to one new repair dispatch, one-hour same-head retries,
  protected `main`, product-level single-flight concurrency, and explicit
  scheduler credentials without direct model secrets or inherited secrets.
- Added permanent caller-contract tests and APA 7th doctoring for default-branch
  activation, immutable reusable-workflow source, failure behavior, rollback,
  and the NVIDIA NIM control-plane boundary.
