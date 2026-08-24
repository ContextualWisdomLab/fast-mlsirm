# Seal enterprise request record admission

## Fixed

- Enterprise issue scoring-request provenance now rejects caller-defined issue, stakeholder-perspective, and candidate-intervention record subclasses before reading their fingerprints or fields, preventing caller callbacks from executing during canonical record admission while preserving exact package record behavior.
