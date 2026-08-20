# Harden IRT readiness integer controls

## Fixed

- Admit only exact built-in and supported concrete NumPy integer scalars for production IRT readiness thresholds, rejecting booleans, caller-defined integer subclasses, and implicit conversion providers before caller coercion/comparison hooks can execute. Existing readiness domains, response-shape rules, and downstream Rust-owned psychometric computation remain unchanged.
