# Rubric testlet pilot control callback hardening

## Scope

Issue #883 hardens only the Python validation/marshalling boundary used by the generated-item testlet pilot handoff. The change does **not** alter testlet likelihood, EM/SQUAREM estimation, quadrature, variance-component arithmetic, convergence, estimands, or Rust numerical ownership.

The protected-main defect admitted caller-dispatchable protocols before trusted scalar identity was established: `operator.index()` could execute arbitrary `__index__` providers for integer controls, broad Python/NumPy numeric subclass admission followed by `float()` could execute caller-defined conversion hooks, and broad string admission followed by `casefold()` could execute caller-defined model-normalization hooks.

## Security rationale

The correction uses positive exact-type admission with identity-only checks. Integer controls accept exact built-in `int` values plus exact package-supported NumPy integer scalar classes. Floating controls accept exact built-in `int`/`float` values plus exact package-supported NumPy integer/floating scalar classes. Boolean controls accept exact built-in `bool` and exact `numpy.bool_`. The model control accepts exact built-in `str` before package-owned case normalization and allowlist validation. Arbitrary protocol objects and caller-defined Python/NumPy/string subclasses are rejected without hashing, equality, conversion, normalization, representation, or native-dispatch callbacks.

This maps to CWE-1287 because the boundary previously validated expected control types too broadly and then performed caller-dispatchable conversion. The accepted implementation follows OWASP ASVS 5.0.0 input-validation guidance while preserving the existing `MAX_MAX_ITER` limit, testlet model vocabulary, Rust-embedded quadrature vocabulary `{7, 11, 15, 21, 31, 41}`, strict booleans, and finite non-negative tolerance/variance-start semantics.

NIST SP 800-218 SSDF 1.1 remains the current final normative SSDF publication. NIST SP 800-218 Rev. 1 / SSDF 1.2 was published as an Initial Public Draft on December 17, 2025; it is tracked here as a draft/watch source rather than represented as final normative guidance.

## Verification contract

Public-handoff regressions exercise `TestletPilotDesign.to_fit_testlet_kwargs()` with arbitrary `__index__` providers, Python/NumPy numeric subclasses, a caller-defined string subclass, and a scalar subclass whose metaclass overrides hashing/equality. Rejected hostile controls must execute zero caller callbacks. Genuine NumPy integer, floating, and Boolean scalar compatibility remains part of the contract.

The test-first predecessor head is retained as lineage only. After implementation changes, CI, security, package, coverage, and review evidence must be regenerated on one unchanged exact head. Automated review evidence does not replace the protected-main independent non-author approval requirement.

## References

MITRE Corporation. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0: V2.2 input validation*. https://owasp.org/www-project-application-security-verification-standard/
