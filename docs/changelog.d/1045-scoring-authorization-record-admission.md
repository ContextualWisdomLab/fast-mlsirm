# Seal scoring engine-authorization record admission

## Fixed

- Reject caller-defined assessment, scoring-request, and engine-descriptor subclasses before authorization policy or provenance fields are read, preserving exact package records, stable validation errors, and existing engine-policy semantics.
