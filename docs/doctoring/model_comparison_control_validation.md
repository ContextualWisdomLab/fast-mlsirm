# Model-comparison control validation

## Decision

Public model-comparison semantic controls are validated as bounded package contracts before any caller-defined representation or numeric conversion hook can execute. `relation` accepts only `ModelRelation` or a built-in `str`; `alpha` and `omega_tol` accept the intended built-in/NumPy real-scalar domain and reject arbitrary coercible objects. Stable package-owned validation messages remain non-reflective.

This hardening changes only the validation boundary. Rust-backed Vuong arithmetic, relation-safe routing, accepted relation identities, thresholds, result fields, and scientific interpretation remain unchanged.

## Security rationale

Python's data model specifies that `str(object)` invokes `object.__str__()`. The built-in `float()` conversion for a general object delegates to numeric conversion methods such as `__float__()`. Performing those conversions on untrusted controls therefore executes caller-defined behavior before a finite vocabulary/range contract has been established. Validation should first prove that the runtime value belongs to the package's accepted scalar/type domain, then normalize trusted values only.

The fail-first regressions use hostile objects whose `__str__`, `__repr__`, or `__float__` raise. Acceptance requires the existing package-owned `ValueError` surfaces without invoking those callbacks and without changing the Rust numerical path.

## Verification contract

- preserve every accepted `ModelRelation` identity and built-in-string relation value;
- preserve accepted built-in and NumPy real scalar semantics for `alpha` and `omega_tol`;
- preserve Boolean, non-finite, probability-range, and non-negative tolerance rejection;
- reject arbitrary caller objects before `str()`, `repr()`, or `float()` hooks execute;
- preserve Rust-owned model-comparison arithmetic and scientific routing;
- require focused regression coverage, full repository CI, Security Scan, SAST, and current-head review before integration.

## Primary technical references — APA 7

Python Software Foundation. (2026). *Data model — Python 3.14.6 documentation*. https://docs.python.org/3.14/reference/datamodel.html

Python Software Foundation. (2026). *Built-in functions — Python 3.14.6 documentation*. https://docs.python.org/3.14/library/functions.html
