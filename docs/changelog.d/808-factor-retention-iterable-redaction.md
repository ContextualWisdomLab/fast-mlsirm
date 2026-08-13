# Factor-retention iterable error redaction

## Fixed

- Governed factor-retention evidence now converts hostile iterator-construction
  and iteration callback failures into stable package-owned validation errors
  without exposing caller-controlled exception text or chained causes.
- Explicit `MemoryError`, duplicate-method precedence, deterministic ordering,
  decision semantics, and the bounded closed-method evidence contract remain
  unchanged.
