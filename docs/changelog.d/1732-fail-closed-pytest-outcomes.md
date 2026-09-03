# Fail-closed pytest outcome accounting

## Fixed

- Make repository pytest evidence non-passing whenever collection or execution records a skipped, expected-failure, or unexpected-pass outcome, so missing Rust/GPU/platform capability cannot be reported as a successful scientific or package gate. Clean all-executed runs retain their original status, and missing terminal outcome accounting fails closed.
- Make descriptor-relative atomic-write capability evidence fail closed when any required POSIX primitive is unavailable instead of returning normally from the three atomic-write tests and recording false passes.
- Enforce the non-execution verdict after ordinary `pytest_sessionfinish` implementations complete, so a later plugin cannot overwrite skipped evidence back to a successful process exit; a pre-existing stronger non-success exit remains non-passing rather than being erased.
