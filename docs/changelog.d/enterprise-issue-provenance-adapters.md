# Governed enterprise issue provenance adapters

## Added

- Added `fast_mlsirm.scoring.enterprise_issue` source, evidence-span, and counterevidence contracts that preserve exact source revisions, bounded offsets, epistemic assertion roles, and issue-statement targets without retaining raw source text or creating a parallel scoring-evidence schema.
- Evidence spans explicitly distinguish directly stated facts, supported inferences, counterevidence, unresolved ambiguity, and stakeholder value judgments, and project into the existing shared `EvidenceReference` boundary without treating sentiment as consequence, likelihood, urgency, priority, or utility.
- Added factory sealing, content addressing, immutable bounded metadata, descriptive identifiers, source-revision replay, stable redacted failures, complete public documentation, APA 7th provenance traceability, and statement/branch coverage for the new adapter boundary.
