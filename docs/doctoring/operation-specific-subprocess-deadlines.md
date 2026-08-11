# Doctoring record: operation-specific subprocess deadlines

## Decision

Repository automation must bound child-process execution without pretending that
all operations have the same expected duration. The ignored Rust statistical-study
runner therefore uses three named deadline classes:

| Operation | Default | Allowed override range |
| --- | ---: | ---: |
| Cargo workspace metadata | 30 seconds | 5–120 seconds |
| Ignored-test inventory | 120 seconds | 30–600 seconds |
| One long statistical study | 1,800 seconds | 60–7,200 seconds |

These values are repository operating policy, not universal performance claims.
The statistical-study deadline is deliberately much longer than the metadata and
inventory deadlines because true-parameter recovery and other ignored scientific
studies are expected to perform substantially more computation. The existing
GitHub Actions job timeout remains a separate outer ceiling; the ignored Rust
shard job currently has a 90-minute job timeout.

## Operator configuration

The three optional environment variables are:

- `FAST_MLSIRM_CARGO_METADATA_TIMEOUT_SECONDS`;
- `FAST_MLSIRM_CARGO_TEST_LIST_TIMEOUT_SECONDS`; and
- `FAST_MLSIRM_STATISTICAL_TEST_TIMEOUT_SECONDS`.

Overrides must be base-10 integer seconds inside the operation-specific bounds.
Booleans, fractional values, scientific notation, empty values, negative values,
and out-of-range values fail closed. Omitting an override uses the documented
default. Passing an explicit environment mapping to the resolver does not
implicitly merge the process environment into that mapping.

## Process boundary and cleanup

The runner passes every command as an argument vector rather than invoking a
shell. On POSIX systems, `subprocess.Popen(..., start_new_session=True)` creates a
new session for the child. When a deadline expires, the package sends `SIGTERM`
to that process group, allows a bounded five-second grace period, escalates to
`SIGKILL` if the group does not exit, and drains the child with `communicate()` so
that the parent does not intentionally leave an unreaped process.

If the process group has already disappeared, the parent still drains the process
handle. On non-POSIX platforms the helper performs a bounded direct-process
`terminate()`/`kill()` fallback. The non-POSIX path is intentionally **not**
documented as providing POSIX-equivalent descendant-tree termination; stronger
platform-specific tree control would require separate evidence.

Python's subprocess documentation defines `TimeoutExpired` as the timeout failure
surface and documents `start_new_session=True` as the POSIX `setsid()` boundary.
It also recommends passing a sequence of program arguments for ordinary process
creation. Those primitives are used here without changing the scientific test
itself.

## Failure evidence and privacy boundary

A timeout raises the package-owned `BoundedSubprocessTimeout`. Its stable
machine-readable projection contains only:

```json
{
  "status": "timeout",
  "operation": "statistical_test",
  "timeout_seconds": 1800.0
}
```

The timeout exception and the ignored Rust runner's JSON stderr record omit the
child command, command arguments, stdout, and stderr. This is deliberate: child
output can contain test data, paths, source-derived values, or other
caller-controlled material and is not necessary to identify the timeout class.
The runner returns exit code `124` for this bounded timeout path and never treats
a timeout as test success. Ordinary non-zero child exits retain the existing
`CompletedProcess`/`CalledProcessError` semantics according to the caller's
`check` setting.

## Scientific boundary

The deadline mechanism does not relax, truncate, sample, or replace any recovery
study. It changes only orchestration around the existing Cargo command. In
particular, it does not turn a timed-out study into evidence of convergence,
parameter recovery, CPU/GPU parity, model fit, reliability, or scientific
validity.

This slice deliberately migrates only `scripts/run_ignored_rust_shard.py`.
Issue #555 remains open for separate, reviewable deadline policies for GitHub CLI,
packaging, release-evidence, ordinary-test, and other repository subprocess call
sites. A single global timeout remains rejected because it would either be too
slow for short metadata commands or too short for legitimate scientific work.

## Verification contract

Permanent tests cover:

1. distinct metadata, inventory, and statistical-study defaults;
2. bounded integer-only operator overrides;
3. malformed command rejection before process creation;
4. POSIX new-session creation;
5. bounded `SIGTERM` then `SIGKILL` process-group escalation;
6. process reaping when a group has already disappeared;
7. non-POSIX terminate/kill fallback without claiming descendant parity;
8. machine-readable timeout evidence that omits child-controlled command/output
   text;
9. preservation of `check=True` non-zero-exit semantics; and
10. integration routing of Cargo metadata, ignored-test inventory, and selected
    statistical tests through the appropriate operation class.

The complete repository CI remains authoritative for Python tests, Rust and PyO3
tests, package/release acceptance, explicit GPU no-skip evidence, fuzzing,
Security Scan, and SAST on the final unchanged head.

## Rollback

Rollback may restore direct subprocess execution, but doing so reintroduces the
unbounded-runner availability risk documented in issue #555. A safer rollback is
to retain the bounded helper and revert only the migrated caller while the failed
operation-specific policy is corrected. Branch protection and GitHub Actions job
timeouts remain independent controls in either case.

## References

GitHub, Inc. (2026). *Workflow syntax for GitHub Actions*. GitHub Docs.
https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax

Python Software Foundation. (2026). *subprocess — Subprocess management*. Python
3.14 documentation. https://docs.python.org/3/library/subprocess.html
