# Release / publish contract (fast-mlsirm)

This document is the contract handed to the ContextualWisdomLab `.github`
lead for centralizing release-tag and PyPI publication. Until one full
release succeeds through the central reusable path, **local workflow bodies
in this repository remain the default working path** (ADR-0029).

## Local entrypoints (keep until central cutover)

| Workflow | Path | Trigger |
|---|---|---|
| Release Tag | `.github/workflows/release-tag.yml` | `workflow_dispatch` only |
| Publish Package | `.github/workflows/publish-pypi.yml` | `workflow_dispatch` only |
| PyPI Gap Guard | `.github/workflows/pypi-gap-guard.yml` | `schedule` (hourly), `workflow_dispatch`, `release: types: [published]` |

**Central** reusable workflows are expected from a forthcoming
`ContextualWisdomLab/.github` PR (owner redirect: `release-tag` +
`publish-package` — note central publish name is `publish-package`, not
necessarily `publish-pypi`). Exact filenames and the merge commit SHA are
TBD; do **not** invent live `uses:` paths in this repository until that PR
merges.

Cutover sequence (separate PRs):

1. Wait for `.github` reusable workflows to merge; record the exact SHA.
2. Open a **separate** fast-mlsirm PR with thin wrappers only:
   `uses: ContextualWisdomLab/.github/.github/workflows/<shipped-file>@<exact-sha>`,
   preserving required check names and immutable-release rules.
3. Keep full local `release-tag.yml` / `publish-pypi.yml` bodies until **one**
   successful end-to-end central release; delete local full copies only after
   that e2e success (not in the thin-wrapper PR).

Local files in **this** PR carry commented dual-path stubs only.

## `release-tag.yml` inputs

| Input | Required | Rule |
|---|---|---|
| `release_version` | yes | Canonical three-component semver without leading zeros (e.g. `0.11.3`). Must equal `project.version` in `pyproject.toml` at `release_commit`. |
| `release_commit` | yes | Full 40-character lowercase SHA-1 of the reviewed version-cut commit. Must be an ancestor of the current default-branch HEAD. |

Dispatch **must** target the repository default branch (`refs/heads/<default>`).

### CHANGELOG rule

Exactly one section whose line starts with:

```text
## [X.Y.Z] -
```

(where `X.Y.Z` is the requested `release_version`). The section body must be
non-empty. Fragment aggregate drift is fail-closed via:

```bash
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

### Release notes cap

Extracted notes are capped at **120_000** characters (GitHub release-body
limit). Oversized sections become a summary with an authoritative
`CHANGELOG.md` link at the release tag.

### Tag / release semantics

- Never overwrite an existing GitHub Release for `vX.Y.Z`.
- An existing tag without a release may resume only when the tag already
  points at `release_commit`.
- Tags are created atomically via the Git refs API at `release_commit`.
- `gh release create` uses `--verify-tag` (never retargets).

### Downstream dispatch

After the release exists, `release-tag` dispatches `publish-pypi.yml` on the
**default branch** with:

| Field | Source |
|---|---|
| `release_tag` | `v$RELEASE_VERSION` |
| `release_commit` | input `release_commit` |
| `control_plane_commit` | `origin/<default-branch>` HEAD at dispatch time |

## `publish-pypi.yml` inputs

| Input | Required | Rule |
|---|---|---|
| `release_tag` | yes | Immutable tag, e.g. `v0.11.3` |
| `release_commit` | yes | Full lowercase SHA-1; must equal `git rev-parse $release_tag^{commit}` and `project.version` must satisfy `v{version} == release_tag` |
| `control_plane_commit` | yes | Full lowercase SHA-1; must equal `github.sha` of the default-branch workflow run |

### Environment and secret

- GitHub Environment: `pypi`
- Secret: `PIPY_TOKEN` (package-owned PyPI token; name is intentional historical spelling)
- Uploader: `pypa/gh-action-pypi-publish` with `skip-existing: true`
- Release-asset attach is isolated from PyPI credentials; immutable releases
  and HTTP 422 already-uploaded assets are skipped so PyPI retries stay green

## Pinned action SHAs (from current local workflow files)

| Action | Pin |
|---|---|
| `actions/checkout` | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `actions/setup-python` | `5fda3b95a4ea91299a34e894583c3862153e4b97` |
| `actions/upload-artifact` | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` |
| `actions/download-artifact` | `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c` |
| `PyO3/maturin-action` | `e83996d129638aa358a18fbd1dfb82f0b0fb5d3b` (`maturin-version: v1.14.1`) |
| `pypa/gh-action-pypi-publish` | `dc37677b2e1c63e2034f94d8a5b11f265b73ba33` |

Central reusable workflows should preserve these pins (or newer reviewed pins)
and the input / environment / secret contract above.

## Gap-guard recovery contract

`pypi-gap-guard.yml` compares GitHub Releases to
`https://pypi.org/pypi/fast-mlsirm/json` (curl; avoid GitHub API spam for
package inventory). For each `vX.Y.Z` release missing on PyPI whose tag
matches `pyproject.toml` at that tag:

1. `release_commit` = `git rev-parse vX.Y.Z^{commit}`
2. `control_plane_commit` = default-branch HEAD
3. Dispatch local `publish-pypi.yml` with those fields
4. Skip versions already on PyPI; never overwrite or retarget tags

## Noema version ownership (forward-looking)

Once the Noema semver gate is live in the central pipeline, do **not**
hand-pick the next version number in fast-mlsirm PRs/cuts. Noema owns
major/minor/patch from changelog fragments + public API inventory diff +
ADR-0028 decisions (fail closed to human).

## Cutover gate

Do **not** delete or hollow out local `release-tag.yml` /
`publish-pypi.yml` bodies until:

1. Central reusable workflows are merged in `ContextualWisdomLab/.github`
   (expected `release-tag` + `publish-package`; exact files/SHA from that PR),
2. A separate thin-wrapper PR pins `uses: ...@<exact-sha>`, and
3. One full release has succeeded end-to-end via the central path.
