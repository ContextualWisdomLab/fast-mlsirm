# Seal enterprise explicit-value integer admission

## Fixed

- Reject caller-defined integer subclasses for enterprise explicit-value source offsets and deterministic parser record limits before comparison or coercion callbacks can execute, while preserving exact built-in integer domains and stable validation errors.
