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
studies are expected to perform substantially more computation. The scheduled
ignored-Rust workflow opts into the full 7,200-second statistical bound and has a
separate 180-minute GitHub Actions shard ceiling; those controls are independent.

The workflow-level override is intentionally limited to the exhaustive study
job. Interactive and other automation callers continue to use the resolver's
documented 1,800-second default unless they supply their own bounded override.

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

The Mokken recovery study keeps the normal-trait condition as the calibrated
recovery contract. Its skew-trait condition is a distribution-sensitivity
control: it requires finite positive ``H`` below the normal-trait value for the
same item pool, but it does not require AISP's conventional ``c = 0.3`` cutoff
to recover every item. Loevinger's ``H`` is a scalability coefficient rather
than a probability, and AISP selection depends on the observed response
distribution; treating that cutoff as distribution-invariant made the hosted
shard fail on a scientifically valid skewed population. This test boundary
follows the Mokken model and item-selection literature rather than masking the
failed study or weakening the Rust implementation.

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

Mokken, R. J. (1971). *A theory and procedure of scale analysis: With
applications in political research*. De Gruyter Mouton.
https://doi.org/10.1515/9783110813203

van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of Statistical
Software, 20*(11), 1–19. https://doi.org/10.18637/jss.v020.i11

Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing optimization
algorithms for item selection in Mokken scale analysis. *Journal of
Classification, 30*(1), 75–99. https://doi.org/10.1007/s00357-013-9122-y
