# Harden enterprise evidence integer boundaries

## Fixed

- Reject caller-defined integer coercion for enterprise source character counts and evidence offsets before any conversion callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility, nonempty-span semantics, and existing bounded `AssessmentSpecError` behavior.
