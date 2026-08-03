# Manual release-tag workflow

## Added

- A fail-closed `workflow_dispatch` release workflow (`release-tag.yml`) that
  verifies the requested semantic version matches `pyproject.toml` and an
  existing CHANGELOG release section, extracts that section as the release
  notes, refuses to overwrite an existing release, and publishes the `v*` tag
  and GitHub Release with a least-privilege job-scoped `contents: write`
  token. Because a `GITHUB_TOKEN`-created tag does not retrigger tag-push
  workflows, the exhaustive Statistical Studies evidence run is dispatched
  separately after a release.
- A release body larger than GitHub's 125,000-character release limit is
  replaced by a bounded summary that links the authoritative CHANGELOG
  section and lists the released feature headings.
- A run interrupted between tag creation and release publication resumes on
  the existing immutable tag: an existing release always blocks, an existing
  tag without a release skips re-tagging and publishes against the verified
  tag, and API uncertainty still fails closed.
