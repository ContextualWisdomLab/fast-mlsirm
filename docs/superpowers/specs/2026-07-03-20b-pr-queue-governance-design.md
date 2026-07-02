# 20B PR Queue Governance Design

## Intent

The KRW 2,000,000,000 buyer-review packet must show that live open PRs are
known, classified, and separated from release evidence. Review delays and
queued checks are not release blockers by themselves, but stale or
changes-requested PRs and release-scope conflicts must be visible to
procurement.

## Scope

Add `scripts/build_pr_queue_governance.py` as an evidence builder. A separate
library, submodule, hosted dashboard, or Figma Code Connect integration is out
of scope because the buyer need is a static evidence artifact tied to the
current repository state.

## Evidence Contract

The script writes:

- `pr_queue_governance_manifest.json` for machine review;
- `pr_queue_governance_report.html` for human procurement review.

The manifest records generated time, source commit, repository, default branch,
open PR count, risk counts, GitHub snapshot mode, and classified PR records.
Each PR classification includes review decision, merge state, update age, stale
status, changes-requested status, duplicate-looking scope, release-scope
conflict status, review/check delay status, URL, head branch, and base branch.

## Risk Categories

- `changes_requested`: reviewer has requested changes.
- `stale`: updated earlier than the configured stale-day threshold.
- `duplicate_candidate`: PR title or branch appears to cover already-productized
  report, CLI, or evidence work.
- `release_scope_conflict`: PR appears to alter model, backend, formula,
  diagnostic, likelihood, or gradient scope.
- `review_or_check_delay`: PR is awaiting review or queued checks.

## Gate Integration

`scripts/build_commercial_release.py` should run this builder by default after
procurement due diligence. `scripts/sales_readiness.py` should accept
`--pr-queue-governance` and `--require-pr-queue-governance` to verify manifest
status, contract value, category coverage, risk-count coverage, failed checks,
and HTML SHA256.

## Non-Goals

This design does not close or merge open PRs, override reviewers, reinterpret
model formulas, add a hosted queue dashboard, or assert that open PR count must
be zero.
