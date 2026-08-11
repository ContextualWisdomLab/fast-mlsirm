# G-theory public-control validation hardening

## Decision

The public G-theory D-study size controls and the `Phi(lambda)` mastery cut are validation inputs, not extension points for arbitrary Python conversion protocols. `fast-mlsirm` therefore validates their scalar type domain before any conversion: positive D-study sizes admit ordinary Python integers and NumPy integer scalars, while the mastery cut admits finite ordinary Python integer/float scalars and NumPy integer/floating scalars. Boolean, non-numeric, non-finite, non-positive, and otherwise unsupported controls fail with stable package-owned `ValueError` messages.

The validation boundary must not call caller-defined `__int__`, `__float__`, or `__repr__` hooks while rejecting unsupported objects. Python remains responsible only for validation and marshalling; the Rust core remains the numerical authority for G-theory ANOVA/variance components, D-study coefficients, and `Phi(lambda)`. This hardening changes neither equations nor score interpretation.

## Evidence

Fail-first source head `000b2ad779d206e4b1511355db4ccd297d4df8b3` reached the installed Rust/PyO3 production boundary in CI run `31446612952`. The complete Python suite finished with exactly three intended failures because the former wrapper executed hostile `__int__`/`__float__` hooks. Rust/PyO3, package/reinstall/release acceptance, enterprise readiness, GPU-no-skip, fuzz, Security Scan, and SAST were otherwise successful.

Product GREEN head `c48140c4a94828234083ecb180b0bfb029123d5a` replaced those coercions with bounded type-domain validators and strengthened regression coverage for hostile callbacks, booleans, zero/negative/fractional sizes, non-finite cuts, strings, and supported Python/NumPy scalar controls. Exact-head CI `31447070678`, Security Scan `31447070686`, and SAST `31447070679` completed successfully before this documentation-only follow-up.

## Interpretation boundary

This is a public-validation and privacy/security control. It does not alter the Huebner–Lucht G-study/D-study estimands, Brennan–Kane `Phi(lambda)` estimator, clamped-ANOVA policy, Rust numerical implementation, or any validity/reliability claim. Existing method-specific scientific references in `fast_mlsirm.gtheory` remain authoritative for the equations.

## References

NumPy Developers. (2026). *Scalars*. NumPy reference. https://numpy.org/doc/stable/reference/arrays.scalars.html

Python Software Foundation. (2026). *Data model*. Python 3.14.6 documentation. https://docs.python.org/3.14/reference/datamodel.html
