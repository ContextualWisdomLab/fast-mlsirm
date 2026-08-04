# Exact task-revision identity for scoring calibration

## Changed

- Scoring-request wire schema `1.1` now requires an exact provider-neutral `task_revision_fingerprint` in addition to the logical task identifier. The fingerprint participates in request identity, is propagated by the essay adapter from the complete prompt fingerprint, and prevents changed task content from being silently pooled under one request or calibration item.
- Criterion-level many-facet handoffs now use exact task revisions as the Rust estimator item axis while retaining aligned logical task and task-family labels for audit. Duplicate cells, support, resource bounds, respondent–item connectedness, item–rater connectedness, and response provenance are all revision-indexed; one revision cannot be rebound to a different logical task or family.
- Added an explicit, fail-closed schema-`1.0` request migration that verifies canonical content, fingerprint, public handle, and the authoritative engine-policy projection; requires a caller-supplied task revision; preserves normalized caller metadata; and intentionally does not migrate legacy observations or results. Content identity prevents accidental pooling but does not establish cross-revision comparability, which still requires anchors, invariance/DIF, drift, and recovery evidence.
