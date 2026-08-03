# Manual release-tag workflow

## Added

- A fail-closed `workflow_dispatch` release workflow (`release-tag.yml`) that
  only accepts dispatches against the repository default branch, verifies the
  requested semantic version matches `pyproject.toml` and exactly one existing
  CHANGELOG release section, extracts that section as the release notes,
  refuses to reuse either an existing tag or an existing release, and publishes
  the `v*` tag and GitHub Release with a least-privilege job-scoped
  `contents: write` token. Release publication is globally serialized. Because
  a `GITHUB_TOKEN`-created tag does not retrigger tag-push workflows, the
  exhaustive Statistical Studies evidence run is dispatched separately after
  a release.
