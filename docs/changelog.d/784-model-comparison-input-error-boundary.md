# Model-comparison hostile input redaction

## Fixed

- Model-comparison parameter counts and casewise iterables redact hostile conversion and iteration callback failures into stable package-owned `ValueError` messages while preserving `MemoryError`.
