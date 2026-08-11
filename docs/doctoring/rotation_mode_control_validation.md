# Rotation mode control validation

## Decision

`rotate_factor_loadings(..., mode=...)` treats `mode` as a finite semantic control, not as a coercible display value. The public boundary accepts `None` or a built-in `str` only. Non-string caller objects are rejected before representation callbacks or Rust rotation work. Package-owned strings retain the existing `orthogonal`/`oblique` vocabulary and aliases.

This is a validation and error-surface hardening decision. It does not change rotation criteria, objective functions, target/weight semantics, optimization, multi-start behavior, CPU/GPU policy, or interpretation.

## Security rationale

Python's data model specifies that `str(object)` calls `object.__str__()` to obtain an object's informal string representation. Consequently, coercing an arbitrary caller object with `str()` executes caller-defined behavior before a finite control vocabulary has been validated. A semantic enum-like API boundary should instead validate the accepted runtime type first and normalize only package-owned string data.

The regression uses an object whose `__str__` and `__repr__` raise. Acceptance requires the package-owned `ValueError("mode must be 'orthogonal' or 'oblique'")` without either callback executing and without reaching the Rust numerical path.

## Verification contract

- preserve `None` default-mode resolution;
- preserve `orthogonal`, `oblique`, `orth`, `t`, `oblq`, and `q` semantics;
- reject non-built-in-string mode values before representation callbacks;
- preserve criterion/mode compatibility checks;
- preserve all accepted numerical results and Rust ownership;
- require focused regression coverage, full repository CI, Security Scan, SAST, and current-head review before integration.

## Primary technical reference — APA 7

Python Software Foundation. (2026). *Data model — Python 3.14.6 documentation*. https://docs.python.org/3.14/reference/datamodel.html

Python Software Foundation. (2026). *Built-in functions — Python 3.14.6 documentation*. https://docs.python.org/3.14/library/functions.html
