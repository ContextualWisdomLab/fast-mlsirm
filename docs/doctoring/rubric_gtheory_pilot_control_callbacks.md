# Rubric G-theory pilot control callback hardening

## Scope

Issue #881 hardens only the Python validation/marshalling boundary used by the generated-item one-facet G-theory pilot handoff. The change does **not** alter G-theory ANOVA, variance-component estimation, D-study coefficients, Phi(lambda), clamping policy, estimands, or Rust numerical ownership.

The protected-main defect admitted caller-dispatchable numeric protocols before trusted scalar identity was established: `operator.index()` could execute arbitrary `__index__` providers for D-study sizes, while broad Python/NumPy numeric subclass admission followed by `float()` could execute caller-defined mastery-cut conversion hooks.

## Security rationale

The correction uses positive type admission. D-study sizes accept exact built-in `int` values plus exact package-supported NumPy integer scalar classes. Mastery cuts accept exact built-in `int`/`float` values plus exact package-supported NumPy integer/floating scalar classes. Booleans, arbitrary integer-like objects, and caller-defined Python/NumPy scalar subclasses are rejected before conversion, representation, comparison, equality, or hashing callbacks are needed.

This maps directly to CWE-1287 because the boundary previously validated an expected numeric type too broadly and then performed caller-dispatchable conversion. The accepted implementation follows OWASP ASVS 5.0.0 input-validation guidance to validate against expected structures, values, and logical limits, while preserving the existing `1..=1_000_000` D-study-size range, 64-row materialization ceiling, and finite-cut contract.

NIST SP 800-218 SSDF 1.1 remains the current final normative SSDF publication. NIST SP 800-218 Rev. 1 / SSDF 1.2 was published as an Initial Public Draft on December 17, 2025; it is tracked here as a draft/watch source rather than represented as final normative guidance.

## Verification contract

Public-handoff regressions exercise `GTheoryPiPilotDesign.to_gtheory_pi_kwargs()` and `to_phi_lambda_kwargs()` with arbitrary `__index__` providers, Python numeric subclasses, NumPy numeric subclasses, and genuine NumPy scalars. Rejected hostile controls must execute zero caller callbacks. Genuine NumPy scalar compatibility and the established resource domains must remain intact.

The test-first predecessor head is retained as lineage only; after implementation changes, CI, security, package, coverage, and review evidence must be regenerated on one unchanged exact head. Automated review evidence does not replace the protected-main independent non-author approval requirement.

## References

MITRE Corporation. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0: V2.2 input validation*. https://owasp.org/www-project-application-security-verification-standard/
