# 20B PR Queue Governance Design

## Intent

The KRW 2,000,000,000 buyer-review packet must show that live open PRs are
known, classified, and separated from release evidence. Review delays and
queued checks are not release blockers by themselves, but stale or
changes-requested PRs, release-scope conflicts, and unresolved duplicate issue
claims must be visible to procurement.

## Scope

`scripts/build_pr_queue_governance.py` builds deterministic static evidence.
A separate library, submodule, hosted dashboard, or Figma Code Connect
integration remains out of scope because the buyer need is an evidence artifact
tied to the current repository state.

## Evidence Contract

The script writes:

- `pr_queue_governance_manifest.json` for machine review;
- `pr_queue_governance_report.html` for human procurement review.

The manifest records generated time, source commit, repository, default branch,
base SHA, open PR count, risk counts, GitHub snapshot mode, active PR heads,
closing issue references, duplicate-claim decisions, changed-file overlap
warnings, and bounded closed/merged claim history. The HTML exposes repository,
base SHA, PR head SHA, issue references, and timestamps without hover-only
content.

## Risk Categories

- `changes_requested`: reviewer has requested changes.
- `stale`: updated earlier than the configured stale-day threshold.
- `duplicate_candidate`: PR title or branch appears to cover already-productized
  report, CLI, or evidence work.
- `release_scope_conflict`: PR appears to alter model, backend, formula,
  diagnostic, likelihood, or gradient scope.
- `review_or_check_delay`: PR is awaiting review or queued checks.

## Duplicate Issue Claim Contract

Closing references are parsed case-insensitively from active PR bodies for
`Closes #N`, `Fixes #N`, and `Resolves #N` forms. Two or more active PRs that
claim the same issue create a blocking conflict unless exactly one claimant
contains an issue-specific marker on its own line:

```text
Canonical-For: #394
```

The marker designates only the named issue. Zero or multiple designations remain
conflicted. Closing or merging a claimant removes it from the active conflict
while preserving the claim in `issue_claim_history`.

Exact duplicate head branches and changed-file Jaccard overlap of at least 0.80
are secondary, non-blocking warnings. File-overlap warnings require each PR to
change at least two files, so a shared central file cannot create a duplicate
warning by itself.

## Gate Integration

`scripts/build_commercial_release.py` runs this builder by default after
procurement due diligence. An unresolved duplicate issue claim makes the
builder fail closed and therefore blocks the commercial release stage. Review
waits, queued checks, duplicate heads, and changed-file overlap warnings remain
evidence rather than automatic release blockers.

The hourly workflow uploads the JSON and HTML with `if: always()` so the failed
gate still leaves inspectable evidence.

## Non-Goals

This design does not close or merge open PRs, override reviewers, reinterpret
model formulas, add a hosted queue dashboard, or claim that open PR count alone
determines release readiness.
