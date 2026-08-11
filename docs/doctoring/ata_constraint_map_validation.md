# ATA semantic-control validation trust boundary

## Status

Implemented on the active `assemble_to_target` branch for exact semantic-control validation before item-information work. This note documents a bounded public-input correction; it does not change ATA item-information or greedy target-assembly arithmetic and does not claim exact test-assembly optimality.

## Problem

The first ATA trust-boundary correction rejected arbitrary `str(...)` / `int(...)` conversion for content/exposure maps, `seed`, and `exposure_max`. Fresh post-integration review found a second finite-domain gap: type-valid but invalid values could still reach psychometric work, and `exclude` still used NumPy integer coercion.

The residual cases were:

- negative `seed` reaching item-information work before NumPy RNG construction rejected it;
- negative content minimum/maximum counts;
- a per-label minimum exceeding its maximum;
- negative exposure counts or exposure item identities outside the calibrated bank;
- Boolean/fractional exclusions silently coercible to integer identities;
- arbitrary exclusion objects able to invoke `__int__`/`__index__`; and
- out-of-bank exclusions being silently ignored by the eligibility loop.

Python's data model makes numeric/string conversion executable behavior: special methods such as `__int__`, `__index__`, and `__str__` may run caller code rather than merely inspect a type (Python Software Foundation, 2026). CWE-1287 is used only as engineering taxonomy for specified-type validation, not as a vulnerability-severity or certification claim (MITRE, 2026).

## Decision

For public ATA semantic controls that are decidable from bank/control metadata before scoring:

- content constraint keys admit only Python `str` / NumPy `str_`;
- content counts admit only exact non-Boolean Python/NumPy integers, must be non-negative, and a shared label's minimum cannot exceed its maximum;
- exposure item identities/counts admit only exact non-Boolean integers, counts must be non-negative, and item identities must exist in the calibrated bank;
- `seed` and `exposure_max` admit only exact non-Boolean integers and must be non-negative;
- `exclude` admits only a one-dimensional NumPy signed/unsigned integer array or an ordinary list/tuple of exact Python/NumPy integers; Boolean, fractional, object, arbitrary-iterable, and hostile integer-like values fail closed without invoking conversion hooks;
- exclusion indices must identify existing bank items rather than being silently ignored; and
- every invalid semantic control above fails with stable package-owned `ValueError` text before `item_information_matrix()` executes.

Once controls are admitted, item information, greedy capped-shortfall selection, content/exposure behavior, deterministic tie-breaking, and Rust-owned numerical semantics remain unchanged (van der Linden, 2005).

## Test-first evidence contract

The branch preserved an exact fail-first run in which package installation and Rust-primary resolution succeeded, the full Python suite reached the public ATA boundary, and the new semantic-range tests alone demonstrated that invalid controls still reached scoring or leaked caller conversion exceptions. GREEN requires, on one unchanged final head:

- negative seed/content/exposure values and impossible per-label min/max relations fail before information work;
- out-of-bank exposure/exclusion identities fail before information work;
- hostile exclusion objects do not execute `__int__` or `__index__`;
- stable error messages do not reflect rejected caller exception text;
- accepted Python/NumPy integer controls and list/tuple/NumPy integer exclusions still reach item-information work;
- equal feasible minimum/maximum content bounds remain accepted;
- ordinary ATA numerical/constraint behavior remains unchanged; and
- Python 3.12/3.14, Rust/PyO3, package/reinstall, existing GPU, fuzz, Security Scan, SAST, coverage/docstring and current-head review gates pass.

## Scope limitations

This correction does not move CAT/test-assembly numerical ownership into Rust, add an exact MIP/solver, alter calibrated item parameters, change target-information equations, or establish consequential-use validity. The separate Rust CAT/test-assembly ownership roadmap remains governed independently.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (CWE version 4.20)*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

Python Software Foundation. (2026). *Data model*. Python 3.14 documentation. https://docs.python.org/3.14/reference/datamodel.html

van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer. https://doi.org/10.1007/0-387-29054-0
