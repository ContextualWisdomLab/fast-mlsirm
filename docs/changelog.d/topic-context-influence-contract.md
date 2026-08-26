# TEPP topic-context posterior admission

## Added

- Add a Rust fail-closed consumer contract for TEPP
  `tepp.topic_context_posterior.v1`, preserving complete posterior draws,
  Event Lineage, temporal provenance, and source-derived BU/PU/team/person
  multiple membership while keeping case-deletion influence unavailable until
  recovery and CPU/GPU parity gates are satisfied. Valid full-data plausible
  values now report the specific missing refit-evidence state; they are never
  relabeled as a case-deleted posterior.
