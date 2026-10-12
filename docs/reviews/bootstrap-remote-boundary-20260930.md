# Bootstrap remote reply boundary

PR #2286 follow-up; verified implementation head:
`5d39cfc3fc3eaa56d9051252cab4adfbb701d434`.

## Defects and corrections

An injected executor could return missing, duplicate, unrelated or malformed
replicate outcomes. Previously the runner converted fields permissively and
wrote each row immediately, so an invalid later row could leave a partially
populated aggregate. It also caught every fit exception as a rejected replicate,
which could hide a missing backend, allocation failure or programming error.

The consumer now validates one outcome per dispatched envelope, run/family,
seed, envelope/input/output identity and reported version/source cohort. Record
IDs and flags are exact types; errors are bounded; arrays have the expected
single-group or multigroup shape. Non-rejected arrays and log likelihood must
contain finite numbers. Rejected arrays use JSON null placeholders. The entire
batch is decoded before any result slot changes.

Only the existing Rust core's four documented unidentifiable-resample error
contracts are classified as statistical rejection: an unobserved category or
an item with no observed responses, in the single-group or free-item multigroup
path. Other exceptions propagate locally and become structured FAILED outcomes
in the subprocess worker. Rejected multigroup parameter arrays retain their
group axis. No estimator or formula changed.

## Runnable evidence

Environment: macOS arm64, CPython 3.14, existing project-local compiled `_core`.
No new installed-wheel, cross-host or GPU execution claim is made.

RED against `da700730`: 20 new boundary cases failed because malformed outcomes
were accepted/partially written or unexpected exceptions were swallowed.

Exact implementation head above, GREEN command:

```sh
.venv/bin/python -m pytest \
  tests/test_bootstrap_remote_boundary.py \
  tests/test_bifactor_bootstrap*.py \
  tests/test_remote_exec*.py \
  tests/test_bootstrap_failure_receipts.py \
  tests/test_no_hidden_pytest_outcomes.py -q -p no:cacheprovider
```

Result: **159 passed, 1 skipped, exit 0**, 44.88 seconds. The skip is the existing
reviewed `tests/test_bifactor_bootstrap_benchmark.py` study-precision benchmark,
which requires `STAGE5_HIGH_Q=1`; it is not additional executed GPU evidence.
The boundary suite has 37 cases, including unexpected exceptions, atomicity,
reordered valid multigroup replies and rejected multigroup null placeholders.

## Remaining acceptance gates

Matching a reported cohort does not attest the actual worker implementation.
Independent worker attestation, truthful effective-device/thread provenance and
installed-wheel two-host/GPU evidence remain separate unfinished #2001 gates.
Local fault-injection tests do not establish those properties, and these results
do not justify closing #2001 or bypassing exact-head hosted review.
