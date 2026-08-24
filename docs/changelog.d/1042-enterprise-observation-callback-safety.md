# Seal enterprise observation admission

## Fixed

- Enterprise issue observation admission now rejects caller-defined scoring-request, evidence-reference, and status-string subclasses before reading provenance or performing enum lookup, preventing caller callbacks during semantic validation while preserving exact package records and serialized status strings.
