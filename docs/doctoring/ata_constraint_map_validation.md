# ATA constraint-map validation trust boundary

## Status

Implemented for the public `assemble_to_target` semantic-control boundary. This note documents a bounded public-input validation correction; it does not change ATA psychometric arithmetic or authorize broader test-assembly optimality claims.

## Problem

`assemble_to_target()` accepted caller-controlled content and exposure maps through unconstrained `str(key)` / `int(value)` conversion after item-information evaluation began. In Python, those conversions invoke `__str__`, `__int__`, and `__index__` special methods. Booleans are also integers under `isinstance`, so `True` could be silently admitted as a seed or exposure ceiling. Therefore conversion of an arbitrary caller object is executable behavior, not a passive type check (Python Software Foundation, 2026).

The public contract did not require arbitrary object-to-string or object-to-integer coercion for content keys, content counts, exposure identities, exposure counts, `seed`, or `exposure_max`. Permitting such coercion before semantic validation created two avoidable risks:

1. caller conversion callbacks could execute or raise outside the package-owned validation surface; and
2. invalid semantic controls could reach psychometric item-information work before rejection.

This is treated as a specified-type validation boundary consistent with CWE-1287: inputs expected to have a particular type should be validated as that type rather than accepted through unconstrained conversion (MITRE, 2026). CWE is used here as engineering taxonomy, not as a vulnerability-severity or certification claim. The correction continues the ATA content-label trust boundary (content elements already string-checked) for the remaining semantic control maps (van der Linden, 2005).

## Decision

For this ATA boundary:

- validate content constraint keys as Python `str` / NumPy `str_` before scoring;
- validate content counts, exposure keys/values, `seed`, and `exposure_max` as exact integers while rejecting `bool` and conversion hooks without invoking `__int__`/`__index__`;
- reject invalid controls with stable package-owned `ValueError` messages before `item_information_matrix` runs; and
- keep information computation, greedy target matching, content minima/maxima semantics, exclusion, exposure ineligibility, tie-breaking, and all Rust-owned numerical semantics unchanged once controls are admitted.

## Evidence contract

GREEN requires all of the following:

- hostile content-map keys raise `content constraint keys must be strings` without `__str__`/`__repr__` execution;
- hostile content-map counts raise `content constraint counts must be integers` without `__int__`/`__index__` execution;
- hostile exposure-map identities raise `exposure_counts keys and values must be integers` without conversion hooks;
- boolean and fractional `seed` / `exposure_max` values raise `{field} must be an integer`;
- invalid controls are rejected before item-information computation; and
- the ordinary ATA suite plus package/Rust/PyO3/GPU/fuzz/security gates remain green.

## Scope limitations

This correction does not re-home CAT/test-assembly numerical ownership into Rust, introduce a MIP solver, or claim optimality for the greedy assembly surrogate. Additional ATA control surfaces receive their own bounded regressions.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (CWE version 4.20)*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

Python Software Foundation. (2026). *Data model*. Python 3.14 documentation. https://docs.python.org/3.14/reference/datamodel.html

van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer. https://doi.org/10.1007/0-387-29054-0
