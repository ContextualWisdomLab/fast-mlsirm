# Hourly bounded PR review-repair caller

## Decision

`fast-mlsirm` owns a schedule-only caller and delegates review-feedback repair to
the organization-owned reusable workflow. The caller does not copy the central
scheduler, OpenCode worker, model configuration, reviewer credentials, approval
logic, or merge policy into the psychometrics repository.

The workflow runs at minute 37 of every hour, scans only
`ContextualWisdomLab/fast-mlsirm` pull requests targeting protected `main`, and
permits at most one new repair dispatch per run. The same exact head cannot be
redispatched more often than once per hour. Product-level and central
single-flight concurrency prevent overlapping maintenance runs.

## Immutable implementation source

The caller uses the reusable workflow at exact commit
`2f16cca4aae2d11ccc928f8e03fdcbd97a96d5a2`. It does not select `main`, another
mutable branch, `HEAD`, or the deprecated `canonical_ref` input. The pinned
central implementation is the reviewed NVIDIA NIM repair plane proposed in
`ContextualWisdomLab/.github#782`.

This product PR must remain Draft until that central commit is integrated into
the protected central default branch, the caller pin is reconciled to the final
accepted immutable commit, and the product caller passes its own exact-head
checks and independent review. A scheduled workflow does not execute from a
pull-request branch; activation begins only after the caller exists on the
repository default branch.

## Credential and model boundary

The caller passes only the established optional scheduler credentials:

- `PR_REVIEW_MERGE_TOKEN`; and
- `OPENCODE_APPROVE_TOKEN`.

It never uses `secrets: inherit`, `COPILOT_GITHUB_TOKEN`, GitHub Models, or a
GitHub token as model authentication. The central workflow owns NVIDIA NIM
model execution and binds `NVIDIA_NIM_API_KEY` only inside the protected repair
worker. The product caller has no model secret and cannot approve, merge,
release, modify branch protection, or broaden the repair worker's file scope.

## Failure and merge behavior

Missing credentials, absent central workflow, stale or malformed PR metadata,
model failure, unresolved review feedback, failed checks, or exact-head drift
must fail closed. The caller may request a bounded repair; it is not merge
evidence. Independent review, current-head required checks, unresolved-thread
protection, latest-pusher rules, and repository branch protection remain the
merge authority.

The minute-37 offset is deliberate. GitHub documents that scheduled workflow
runs may be delayed during high load, especially at the start of the hour, and
recommends scheduling at another minute. GitHub also specifies that scheduled
workflows run only from the latest commit on the default branch.

## Verification

Permanent contract tests assert:

1. exact hourly cron and absence of branch-selected/manual triggers;
2. immutable central workflow SHA and absence of mutable refs;
3. exact target repository and base branch;
4. one repair dispatch and one-hour same-head retry bounds;
5. product-level single-flight concurrency;
6. least-privilege permissions;
7. explicit scheduler-secret forwarding only; and
8. absence of `secrets: inherit`, `COPILOT_GITHUB_TOKEN`, and direct model
   credentials.

## Rollback

Delete the product caller or pin it to a previously reviewed central commit.
Rollback must not replace immutable delegation with copied privileged code or a
mutable central branch. Manual review, repository checks, and branch protection
continue to operate when the caller is absent.

## References

GitHub. (2026). *Events that trigger workflows*. GitHub Docs.
https://docs.github.com/en/enterprise-cloud@latest/actions/reference/workflows-and-actions/events-that-trigger-workflows

GitHub. (2026). *Reusing workflow configurations*. GitHub Docs.
https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows

GitHub. (2026). *Workflow syntax for GitHub Actions*. GitHub Docs.
https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
