# 20B PR Queue Governance Implementation Plan

## Goal

Make live open PR risk reviewable as part of the commercial evidence bundle,
without changing model formulas, estimator scope, package layout, or Figma Code
Connect status.

## Steps

1. Add `scripts/build_pr_queue_governance.py`.
   - Read open PRs from `gh pr list` in live mode.
   - Read `--offline-snapshot` JSON in fixture mode.
   - Fail offline mode without a snapshot so the fixture requirement is
     explicit.
   - Emit manifest and CSP-protected standalone HTML report.
2. Classify each PR.
   - Record review decision, merge state, head/base branch, URL, update age,
     stale flag, changes-requested flag, duplicate-looking flag, release-scope
     conflict flag, and review/check-delay flag.
   - Treat review/check queue delays as tracked evidence rather than blockers.
3. Integrate the commercial release builder.
   - Run PR queue governance after procurement due diligence by default.
   - Add manifest and HTML artifacts to the commercial release manifest.
   - Provide skip and offline-snapshot options.
4. Extend sales-readiness validation.
   - Add `--pr-queue-governance`.
   - Add `--require-pr-queue-governance`.
   - Validate status, contract value, failed checks, category coverage, risk
     count coverage, open PR count, HTML existence, and HTML SHA256.
5. Update buyer evidence documentation.
   - README.
   - Commercial readiness.
   - Enterprise sales readiness.
   - Release acceptance guide.
   - 20B product readiness.
   - Buyer storyboard.
   - Figma product design packet.
   - ROI evidence model.
   - Enterprise demo manifests.
6. Verify.
   - Targeted tests for PR queue governance, commercial builder integration,
     and sales readiness validation.
   - Full Python and Rust test suites.
   - Package build and `twine check`.
   - Live PR queue smoke.
   - Final sales readiness gate with PR queue governance required.
7. Publish.
   - Commit and push `codex/20b-pr-queue-governance`.
   - Create PR.
   - Inspect live PR/check/review state.
   - Merge after required evidence is clean, treating review/check queue delay
     as a process delay rather than an implementation blocker.
