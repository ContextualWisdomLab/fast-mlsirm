# Exact task-revision calibration identity

## Changed

- Breaking: governed `ScoringRequest` values now require an exact
  `task_revision_fingerprint`, so calibration artifacts distinguish task
  revisions instead of silently pooling responses collected under
  different task content, and the facets calibration handoff replays the
  task-revision provenance fail-closed.
- A `migrate_scoring_request_v1` migration rebuilds verified legacy
  requests under an explicit caller-supplied task revision after
  replaying the legacy identity; unverifiable or mutated legacy
  artifacts are rejected rather than silently upgraded.
