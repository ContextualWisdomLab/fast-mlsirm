# Harden essay adapter integer boundaries

## Fixed

- Reject caller-defined integer coercion across essay prompt limits, submission counts, and evidence offsets before any conversion callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError` semantics.
