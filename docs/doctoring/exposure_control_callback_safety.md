# Exposure-control scalar callback safety

## Scope

The public CAT/exposure wrappers accept integer cardinalities, indices, simulation counts, seeds, stage counts, and related control values. These values are validation/marshalling controls; they do not own psychometric or statistical arithmetic. Selection, routing, scoring, posterior updates, recovery, simulation, and exposure calculations remain Rust-owned.

Protected-main behavior previously used broad `isinstance(..., (int, numpy.integer))` / `isinstance(..., (float, numpy.floating))` acceptance in `_as_int`, then normalized accepted values through operations such as `int(...)`, finite checks, equality, and range comparisons. Caller-defined numeric subclasses can override conversion or comparison behavior, so nominal subtype acceptance could execute caller-controlled callbacks before a package-trusted integer had been established. Several public wrappers also discovered the compiled core before validating their integer controls, making validation order dependent on native-extension availability.

The corrected boundary uses identity-only allowlisting. Exact built-in Python integers are accepted directly; exact supported NumPy integer scalar classes are normalized with `int(...)`; exact built-in and supported NumPy floating scalars remain accepted only when finite and integral. Booleans, numeric subclasses, and other objects are rejected before caller-dispatchable conversion, representation, equality, or range callbacks. Bounds are evaluated only after normalization to a built-in integer. Affected public wrappers establish their integer controls before `from . import _core`, so invalid controls fail with package-owned validation errors independently of compiled-core discovery.

No likelihood, Fisher/KL information, posterior, item-selection, exposure-calibration, routing, scoring, uncertainty, recovery, or simulation formula changes in this lane. The change is limited to Python boundary validation and marshalling, preserving the repository's Rust-first production numerical ownership contract.

## Security and reliability interpretation

MITRE CWE-1287 describes failures to validate that input is actually of the specified type and recommends validating expected input properties rather than relying on detection of particular malformed values. The identity allowlist here follows that accept-known-good principle for the small set of scalar control types this API intentionally supports.

NIST SP 800-218, Secure Software Development Framework (SSDF) Version 1.1, remains the final SSDF baseline. NIST SP 800-218 Rev. 1 / SSDF Version 1.2 is an Initial Public Draft as of this record and is treated as draft guidance, not represented as a final standard. OWASP ASVS 5.0.0 is the latest stable ASVS release; its input-validation guidance is informative for this caller-controlled boundary, but `fast-mlsirm` is a numerical library and this change does not claim ASVS conformance or certification.

## Verification contract

Regression and hosted evidence for this boundary must establish all of the following:

- exact built-in Python integers and genuine supported NumPy integer scalar types remain accepted;
- finite integral built-in and genuine supported NumPy floating scalars preserve the established compatibility contract;
- booleans, non-integral/non-finite floats, out-of-range values, and unsupported objects fail with package-owned errors;
- caller-defined built-in-integer, NumPy-integer, and floating subclasses are rejected before hostile conversion or representation callbacks execute;
- affected CAT/exposure public entry points validate integer controls before native-core discovery;
- valid-control numerical behavior remains delegated to the Rust core without duplicated Python arithmetic; and
- the unchanged exact PR head must pass applicable Python/Rust tests, coverage, security, package, SBOM/provenance, and review gates before readiness or protected-main integration.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (Version 4.20).* Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0.* https://owasp.org/www-project-application-security-verification-standard/
