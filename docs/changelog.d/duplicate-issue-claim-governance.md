# Fail-closed duplicate issue-claim governance

## Added

- PR queue governance now parses `Closes`, `Fixes`, and `Resolves` references
  from every active pull-request body and blocks the commercial release gate
  when multiple open PRs claim one issue without exactly one issue-specific
  `Canonical-For: #N` designation.
- Deterministic JSON and accessible standalone HTML evidence now include the
  repository base SHA, active PR head SHAs, issue references, timestamps,
  duplicate-head warnings, high changed-file-overlap warnings, and bounded
  closed/merged claim history.
- One-file intersections are excluded from changed-file duplicate warnings, and
  the hourly read-only workflow uploads governance artifacts even when the
  duplicate-claim gate fails.
