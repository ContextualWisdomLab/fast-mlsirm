# Harden scoring-policy integer callback boundaries

## Fixed

- Reject caller-defined integer coercion at scoring-policy positive-integer boundaries before any `__index__` callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError` semantics.
