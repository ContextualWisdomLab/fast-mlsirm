# Rubric hostile iterable error redaction

## Fixed

- Rubric collection materialization fails closed on hostile iterable setup and iteration exceptions with package-owned messages, while preserving `MemoryError` resource signals.
