# Harden governed scoring execution integer boundaries

## Fixed

- Reject caller-defined integer coercion at governed scoring request, observation, and result controls before any `__index__` callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError` semantics.
