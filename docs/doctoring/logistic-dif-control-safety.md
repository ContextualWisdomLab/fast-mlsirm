# Observed-score logistic DIF control trust boundary

## Decision

The public observed-score logistic DIF and purification adapters validate and normalize semantic scalar controls before any caller-owned response/group materialization and before compiled-core discovery. Only exact built-in scalar identities and explicitly supported concrete NumPy scalar identities are admitted; caller-defined subclasses and arbitrary conversion-protocol providers are rejected without executing their conversion callbacks.

This is a trust-boundary change, not a statistical-method change. Logistic-regression likelihood/IRLS computation, Mantel-Haenszel statistics, purification sweeps, Benjamini-Hochberg decisions, effect sizes, and returned result structure remain in the Rust implementation.

## Native-domain alignment

The Python boundary mirrors the existing native contract rather than inventing new estimator semantics:

- `exclude_studied_item` is an exact Boolean control;
- `fdr_q` is finite and in `(0, 1]`;
- `max_iter` and `max_rounds` are positive Rust `usize` controls;
- `min_anchor_items` is a nonnegative Rust `usize`; zero remains admissible because the native purification guard applies a minimum effective anchor count of one.

Values outside the native `usize` range fail before PyO3 conversion. Genuine NumPy integer and floating scalar identities remain accepted for numeric controls so previously supported numerical caller code does not lose interoperability.

## Threat model and verification

The defect class is improper validation of the specified input type: invoking `bool(...)`, `float(...)`, or `int(...)` on an untrusted object can execute caller-owned Python code before the library establishes the intended control type/range. MITRE CWE-1287 recommends an accept-known-good strategy that validates type and relevant value properties before use. The regression suite therefore uses hostile subclasses/protocol providers whose conversion hooks increment counters or raise, plus data/core sentinels that prove rejected controls terminate before either caller data or native discovery is touched.

The implementation intentionally does not duplicate logistic, MH, purification, or FDR formulas in Python. Statistical parity remains the responsibility of the existing Rust-backed tests; this slice verifies admission, ordering, exact normalized payload types, and native-domain fidelity.

## References

French, A. W., & Maller, S. J. (2007). Iterative purification and effect size use with logistic regression for differential item functioning detection. *Educational and Psychological Measurement, 67*(3), 373–393. https://doi.org/10.1177/0013164406294781

MITRE. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic regression procedures. *Journal of Educational Measurement, 27*(4), 361–370. https://doi.org/10.1111/j.1745-3984.1990.tb00754.x
