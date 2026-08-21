# Harden rubric text schema callback safety

## Fixed

- Harden rubric, item-blueprint, and shared scoring text/identifier schema admission so caller-defined `str` subclasses fail closed before any overridable text callback executes, while preserving normalization for exact built-in strings.
- Apply the same exact-built-in-string admission to item-bank evidence enums so lifecycle evidence cannot dispatch caller-defined equality or hash callbacks during enum lookup.
