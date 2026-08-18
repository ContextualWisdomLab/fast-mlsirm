# Harden rubric text schema callback safety

## Fixed

- Harden rubric, item-blueprint, and shared scoring text/identifier schema admission so caller-defined `str` subclasses fail closed before any overridable text callback executes, while preserving normalization for exact built-in strings.
