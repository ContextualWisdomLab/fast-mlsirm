# ADR-0029: Central release/publish dual-path and PyPI gap guard

Status: Proposed
Date: 2026-09-18
Supersedes: none
Superseded by: none

## Context

`fast-mlsirm` owns a fail-closed manual release-tag workflow and an explicit
`publish-pypi.yml` control plane (environment `pypi`, secret `PIPY_TOKEN`).
Org-level reusable workflows in `ContextualWisdomLab/.github` are intended to
absorb that contract, but they are not yet published with confirmed paths and
pinned refs. A prior release can also leave a GitHub Release without a matching
PyPI version when the publish dispatch fails, which needs an idempotent recovery
path that does not retarget immutable tags.

## Decision drivers

- Keep shipping releases without waiting on unpublished org reusable workflows.
- Hand the `.github` lead an exact, copyable contract (inputs, CHANGELOG rule,
  notes cap, dispatch fields, pins, environment/secret).
- Recover missing PyPI publications without overwriting tags or inventing
  versions.
- Avoid enabling a live `uses:` caller that would fail until central workflows exist.

## Ownership and dependency direction

`ContextualWisdomLab/fast-mlsirm` remains the package owner and local
release/publish authority until cutover. `ContextualWisdomLab/.github` will own
reusable workflow definitions once published. This ADR does not reverse the
fast-mlsirm vs Psychometrics Commons product boundary.

## Decision

1. Keep the full local bodies of `.github/workflows/release-tag.yml` and
   `.github/workflows/publish-pypi.yml` as the **default working path**.
2. Document the contract in
   `docs/orchestration/release-publish-contract.md` and leave **commented**
   dual-path stubs only. Expected central names from the `.github` lead are
   `release-tag` and `publish-package` (align to whatever
   `ContextualWisdomLab/.github` actually ships). Exact SHA/path is TBD; do
   not invent a live `uses:` that would fail. Thin wrappers pinned at
   `@<exact-merged-sha>` land in a **separate** PR after that merge.
3. Add `.github/workflows/pypi-gap-guard.yml` to detect GitHub Release versions
   missing from `https://pypi.org/pypi/fast-mlsirm/json` and dispatch
   `publish-pypi.yml` with `release_tag`, `release_commit` (from the tag), and
   `control_plane_commit` (default-branch HEAD).
4. Treat immutable / HTTP 422 already-uploaded GitHub release assets as
   successful skips so PyPI-only retries remain green.
5. Do **not** remove local workflow bodies until one full release succeeds via
   the central path.
6. Once Noema's central semver gate is live, do not hand-pick the next version
   number in fast-mlsirm cuts; Noema owns major/minor/patch from changelog
   fragments + public API inventory diff + ADR-0028 (fail closed to human).

## Invariants / acceptance evidence

1. Local `release-tag.yml` / `publish-pypi.yml` remain executable
   `workflow_dispatch` entrypoints with the documented inputs.
2. Gap-guard only dispatches for `vX.Y.Z` tags that match `pyproject.toml` at
   that tag and are absent from PyPI JSON.
3. Gap-guard never creates or moves tags.
4. Contract doc lists the pinned action SHAs currently used by the local
   publish/release workflows.
5. Existing workflow contract tests in `tests/test_release_tag_workflow.py` and
   `tests/test_publish_pypi_workflow_contract.py` remain green.

## Non-goals and claims not made

- This ADR does not publish org reusable workflows.
- This ADR does not weaken the central Security Scan gate.
- This ADR does not authorize force-push, tag retargeting, or PyPI overwrites.
- This ADR does not recreate hosted product release orchestration inside
  fast-mlsirm beyond package publication.

## Consequences and trade-offs

### Benefits

- Releases continue on the known-good local path.
- Centralization work has a written handoff contract.
- Missed PyPI publishes can self-heal without operator guesswork.

### Costs / risks

- Dual documentation must stay in sync until cutover.
- Gap-guard can dispatch republish attempts; mitigated by `skip-existing: true`
  and tag immutability checks in `publish-pypi.yml`.

## Alternatives considered

### Alternative A — Switch to central `uses:` immediately

Rejected: org paths/SHAs are not published; a live caller would break release.

### Alternative B — Docs-only, no gap-guard

Rejected: leaves release/PyPI divergence as a manual recovery problem.

### Alternative C — Delete local workflows once stubs exist

Rejected: violates the one-successful-central-release cutover gate.

## Failure, degraded, and recovery behavior

- Gap-guard skips versions already on PyPI (idempotent).
- Non-canonical tags are skipped; mismatched tag/pyproject pairs fail closed.
- Publish retries tolerate already-uploaded PyPI filenames and GitHub asset 422s.
- If central reusable workflows later diverge, keep local bodies until a full
  central release succeeds, then cut over in a dedicated PR.

## Security and privacy implications

- Gap-guard needs `actions: write` only to dispatch `publish-pypi.yml`.
- PyPI credentials remain scoped to environment `pypi` / secret `PIPY_TOKEN`.
- No new long-lived tokens; no Security Scan weakening.

## Compatibility, migration, and rollback

- Migration to central callers is additive and commented until cutover.
- Rollback is leaving local bodies enabled (current default).
- Removing local bodies is explicitly out of scope for this ADR's acceptance.

## Verification and release evidence

- Workflow+docs PR against `main`.
- Local release/publish contract tests pass.
- Manual `workflow_dispatch` of gap-guard is safe when PyPI is complete (no-op).

## Research and standards basis

Not applicable; release/publish governance is operational, not psychometric.

## Follow-ups

- Wait for `ContextualWisdomLab/.github` PR that lands reusable
  `release-tag` + `publish-package` workflows; record the merge SHA.
- Open a separate fast-mlsirm thin-wrapper PR pinning
  `uses: ...@<exact-sha>` (ask again if paths/SHA unknown at that time).
- After one successful e2e central release, a later PR deletes local full
  workflow copies.
- Wire Noema semver ownership once that central gate is live.

## Reversal / supersession conditions

Supersede when central reusable workflows are the sole release/publish path and
local bodies have been removed after evidenced cutover, or when the org ships
different reusable filenames than the expected `release-tag` /
`publish-package` pair.
