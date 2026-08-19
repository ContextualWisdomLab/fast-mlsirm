### Fixed

- Assessment assembly now rejects `ConstructSpec`, `RubricSpecification`, and scoring-policy subclasses before reading package-owned provenance or construct-scope fields, preventing caller-defined attribute/fingerprint callbacks from executing during aggregate contract admission while preserving exact package records and existing cross-reference semantics.
