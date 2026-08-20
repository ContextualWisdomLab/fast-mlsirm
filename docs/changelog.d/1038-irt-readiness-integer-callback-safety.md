# Harden IRT readiness integer callback boundaries

## Fixed

- Reject caller-defined integer subclasses and conversion providers at IRT experiment-readiness controls before caller callbacks can run, while preserving exact built-in and concrete NumPy integer scalar compatibility and existing readiness domains/errors.
