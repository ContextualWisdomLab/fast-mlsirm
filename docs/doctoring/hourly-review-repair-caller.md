# Hourly bounded PR review-repair caller — retired

**Status: SUPERSEDED**

## Decision

The repository-local hourly review-repair caller introduced by PR #763 is retired.
`fast-mlsirm` now preserves a single writer plane and does not schedule a second
repository workflow that can dispatch review-feedback mutations independently of
the dedicated repository writer. The organization-owned `.github` repository
remains a read-only governance dependency from this repository; it is not a
second `fast-mlsirm` writer.

This is a control-plane correction, not a relaxation of review or merge gates.
Required CI, security checks, unresolved-thread handling, exact-head evidence,
independent approval where live governance requires it, and protected-branch
rules remain authoritative.

## Empirical failure evidence

The caller never demonstrated an operational scheduled run on protected `main`.
Its first two default-branch scheduled executions both ended in
`startup_failure` before GitHub created any jobs:

- run `31531589790` (run number 1); and
- run `31537656804` (run number 2).

Both runs referenced the immutable central reusable-workflow commit
`2f16cca4aae2d11ccc928f8e03fdcbd97a96d5a2`. Because the failure occurred before
job creation, rerunning product code or changing psychometric tests cannot alter
the failing boundary. More importantly, retaining a second scheduled mutation
plane conflicts with the repository's single-writer lease even if the reusable
workflow were later repaired upstream.

## Replacement contract

The repository therefore enforces these invariants:

1. `.github/workflows/hourly-review-repair.yml` is absent;
2. no repository-owned workflow delegates to `pr-review-fix-scheduler.yml@...`;
3. repository code, tests, docs, refs, and PR state are mutated through one
   active `fast-mlsirm` writer lease at a time; and
4. review and merge authority remain separate from writer identity.

`tests/test_hourly_review_repair_workflow.py` is retained as a regression guard
for the *absence* of the competing caller so a path rename cannot silently
restore the retired scheduler.

## Lineage

- PR #558 proposed the original caller and closed unmerged.
- PR #763 reapplied and merged the caller onto protected `main`.
- The post-merge scheduled evidence above showed two startup failures with zero
  jobs, and the single-writer control-plane contract supersedes the design.

Historical changelog entries remain historical evidence that the caller existed;
they must not be interpreted as current architecture or operational capability.

## Rollback / reintroduction boundary

Do not restore the repository-local scheduler merely to retrigger review or work
around approval latency. Reintroduction requires a new explicit architecture and
governance decision that proves there is still exactly one mutation writer for
`fast-mlsirm`, demonstrates an operational default-branch run, preserves
independent reviewer identity, and does not weaken required checks or branch
protection.
