# ATA content-label validation trust boundary

## Status

Implemented on active PR #682 only until protected-main integration. This note documents a bounded public-input validation correction; it does not change ATA psychometric arithmetic or authorize broader test-assembly claims.

## Problem

`assemble_to_target()` accepts caller-controlled content labels that are used only as a finite symbolic vocabulary for content constraints. The predecessor implementation converted an object array to strings with `astype(str)` after item-information evaluation. In Python, `str(object)` invokes the object's `__str__()` special method, whose default behavior can delegate to `__repr__()`. Therefore conversion of an arbitrary caller object is executable behavior, not a passive type check. Python 3.14 documents this dispatch explicitly (Python Software Foundation, 2026).

The public contract did not require arbitrary object-to-string coercion. Permitting such coercion before semantic validation created two avoidable risks:

1. caller representation callbacks could execute or raise outside the package-owned validation surface; and
2. invalid content controls could reach psychometric item-information work before rejection.

This is treated as a specified-type validation boundary consistent with CWE-1287: inputs expected to have a particular type should be validated as that type rather than accepted through unconstrained conversion (MITRE, 2026). CWE is used here as engineering taxonomy, not as a vulnerability-severity or certification claim.

## Decision

For this ATA boundary:

- validate the content container shape against the calibrated item count before item-information evaluation;
- accept only Python `str` and NumPy `str_` element values;
- reject any other element type with a stable package-owned `ValueError` without calling its `__str__()` or `__repr__()`;
- only after that type check, normalize accepted string scalars to the package-owned NumPy string representation used by existing content-constraint logic; and
- keep information computation, greedy target matching, content minima/maxima, exclusion, exposure, tie-breaking, and all Rust-owned numerical semantics unchanged.

NumPy documents `ndarray.astype()` as a value cast to another data type; it is therefore appropriate only after the semantic type boundary has admitted values whose conversion is part of the package contract, not as the mechanism for deciding whether arbitrary objects are valid labels (NumPy Developers, 2026).

## Evidence contract

The fail-first regression uses a label object whose `__str__()` and `__repr__()` raise a unique sentinel and replaces item-information computation with a call counter. Valid RED requires the public ATA function to reach the predecessor conversion and expose the callback execution; setup/import/fixture failure is not evidence.

GREEN requires all of the following:

- hostile non-string labels raise the stable package validation error;
- neither representation callback executes;
- invalid content is rejected before item-information computation;
- malformed content shape is rejected before item-information computation;
- NumPy string scalars remain accepted; and
- the ordinary ATA suite plus package/Rust/PyO3/GPU/fuzz/security gates remain unchanged and green.

## Scope limitations

This correction does not establish that every ATA control has completed hostile-object review. It does not change the separate Rust-first migration for CAT/test-assembly numerical ownership, introduce a new optimizer, or make an optimality claim for the existing greedy assembly method. Any additional caller-control coercion finding receives its own bounded regression and review.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (CWE version 4.20)*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

NumPy Developers. (2026). *numpy.ndarray*. NumPy reference. https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

Python Software Foundation. (2026). *Data model: `object.__str__`*. Python 3.14 documentation. https://docs.python.org/3.14/reference/datamodel.html#object.__str__
