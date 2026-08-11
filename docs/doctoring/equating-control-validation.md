# Equivalent-groups control validation doctoring

## Decision

The public Python wrapper for equivalent-groups observed-score equating validates
control values before it loads or calls the compiled Rust boundary. The method
validator accepts only the existing Rust vocabulary (`mean`, `linear`, and
`equipercentile`, including their documented aliases and case/separator
variants). Explicit score ceilings accept only built-in Python integers and
genuine NumPy integer scalars, and they must be positive. Booleans, fractional
values, non-positive values, and arbitrary objects are rejected with a
field-specific `ValueError`.

This is a boundary-validation change. It does not change the Rust equations,
score-array conversion, inferred-ceiling behavior, or the NEAT, kernel,
log-linear, small-sample, and nominal-equating controls. Those controls require
separate contracts and follow-up validation slices.

## Defect and threat model

Before this change, `equate_observed_scores` called `str(method)` and the shared
ceiling helper called `int(k)`. A caller could therefore provide an object whose
`__str__`, `__repr__`, `__int__`, or `__index__` implementation executed code,
raised an unrelated exception, or consumed unbounded resources before
package-owned validation. This violated the fail-closed input contract and made
the Python-to-Rust boundary dependent on caller callbacks.

The implementation now performs exact type checks, normalizes only trusted
values, checks the supported method vocabulary and positive ceiling invariant,
and then passes the normalized controls to Rust. No error-path formatting uses
the hostile object itself.

## Accepted representation contract

| Control | Accepted values | Rejected values |
| --- | --- | --- |
| `method` | Built-in `str` values recognized by the equivalent-groups Rust parser: `mean`/`m`, `linear`/`lin`/`l`, and `equipercentile`/`equip`/`ep`, with existing case and `-`/`_` normalization | Non-strings, unsupported strings, and objects with custom representation methods |
| `k_x`, `k_y` | Positive built-in Python `int` or positive `numpy.integer` scalar | `bool`, zero, negative integers, fractional values, strings, and arbitrary conversion objects |
| omitted ceiling | `None`, followed by the existing finite-data inference path | Empty or non-finite score vectors when inference is required |

## Regression evidence

The focused contract tests prove that:

1. a hostile method object cannot execute `__str__` or `__repr__` and cannot
   reach the Rust stub;
2. hostile explicit ceilings cannot execute `__int__`, `__index__`, or
   `__repr__` and cannot reach the Rust stub;
3. unsupported method strings fail before Rust dispatch;
4. booleans, fractional, zero, negative Python, and negative/zero NumPy integer
   ceilings fail before Rust dispatch; and
5. genuine NumPy integer ceilings still normalize to the same successful Rust
   call and result shape.

The production module retains docstrings for the public and boundary helpers.
The changed production file is required to retain complete statement and branch
coverage in the repository's focused coverage gate.

## Verification and rollback

Verification is ordered from the smallest boundary test to the package and Rust
validation gates: focused Python tests, changed-file lint and docstring checks,
changed-file branch coverage, editable PyO3 import, Rust workspace tests, and
the hosted PR Checks at the exact final commit. A rollback removes the wrapper
validation commit only; the Rust equating implementation and its scientific
method remain unchanged.

## References (APA 7th ed.)

Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
Methods and practices* (3rd ed.). Springer. https://doi.org/10.1007/978-1-4939-0317-7

MITRE. (n.d.). *CWE-20: Improper input validation*. Retrieved August 11, 2026,
from https://cwe.mitre.org/data/definitions/20.html

National Institute of Standards and Technology. (2022). *Secure software
development framework (SSDF) version 1.1: Recommendations for mitigating the
risk of software vulnerabilities (NIST SP 800-218).* https://doi.org/10.6028/NIST.SP.800-218

NumPy developers. (n.d.). *Array scalars*. NumPy documentation. Retrieved August
11, 2026, from https://numpy.org/doc/stable/reference/arrays.scalars.html
