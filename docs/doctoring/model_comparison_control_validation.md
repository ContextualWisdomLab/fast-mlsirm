# Model-comparison control validation

## Decision

Public model-comparison semantic controls are validated as bounded package contracts before any caller-defined representation or numeric conversion hook can execute. `relation` accepts only `ModelRelation` or a built-in `str`; `model_a` and `model_b` accept exact built-in strings before normalization; `k_a` and `k_b` accept exact built-in integers or genuine supported NumPy integer scalar classes; and `alpha` and `omega_tol` accept exact built-in real values or genuine supported NumPy integer/floating scalar classes. Caller-defined Python/NumPy subclasses and arbitrary integer-protocol providers are rejected before `__index__`, `__int__`, `__float__`, string-normalization, or representation callbacks can execute. Stable package-owned validation messages remain non-reflective.

This hardening changes only the validation boundary. Rust-backed Vuong arithmetic, relation-safe routing, accepted relation identities, thresholds, result fields, and scientific interpretation remain unchanged.

## Security rationale

Python's data model permits general integer conversion protocols such as `__index__` and numeric conversions such as `__float__`; string subclasses may also override normalization methods. Invoking these protocols before the package establishes a trusted scalar or label identity executes caller-controlled behavior inside a validation boundary. Module-name or inheritance metadata is not sufficient evidence that a scalar instance is one of the genuine NumPy scalar classes intended by the public contract.

Validation therefore admits only exact built-in control types and an explicit finite set of genuine NumPy scalar classes, then normalizes those already-trusted values. This keeps user-defined subclasses and arbitrary protocol providers outside the trusted conversion boundary without changing the Rust numerical owner.

The fail-first regressions use hostile `__index__`, integer-subclass, string-subclass, and spoofed NumPy floating-subclass controls. Acceptance requires the existing package-owned `ValueError` surfaces with zero hostile callback executions while ordinary supported NumPy scalars continue to normalize successfully.

## Verification contract

- preserve every accepted `ModelRelation` identity and built-in-string relation value;
- preserve exact built-in model labels while rejecting caller-defined string subclasses before normalization;
- preserve exact built-in integers and genuine supported NumPy integer scalars for non-negative parameter counts;
- preserve accepted exact built-in and genuine NumPy real-scalar semantics for `alpha` and `omega_tol`;
- preserve Boolean, non-finite, probability-range, non-negative count, and non-negative tolerance rejection;
- reject arbitrary integer-protocol providers and caller-defined Python/NumPy scalar subclasses before conversion or representation hooks execute;
- preserve Rust-owned model-comparison arithmetic and scientific routing;
- require focused regression coverage, full repository CI, Security Scan, SAST, and current-head review before integration.

## Primary technical references — APA 7

Python Software Foundation. (2026). *Data model — Python 3.14.6 documentation*. https://docs.python.org/3.14/reference/datamodel.html

Python Software Foundation. (2026). *Built-in functions — Python 3.14.6 documentation*. https://docs.python.org/3.14/library/functions.html
