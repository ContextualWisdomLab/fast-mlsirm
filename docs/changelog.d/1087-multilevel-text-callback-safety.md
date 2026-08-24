# Harden multilevel text callback safety

## Fixed

- Require exact built-in strings for contextual schema versions, descriptive identifiers, and provenance fingerprints before comparison, normalization, regex, or encoding work, preventing caller-defined `str` subclasses from executing callbacks during multilevel and temporal contract admission.
