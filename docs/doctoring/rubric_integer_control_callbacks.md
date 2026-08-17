# Rubric integer control callback hardening

## Scope

Issue #891 hardens only the Python validation/marshalling boundary for rubric-centered authoring schemas in `fast_mlsirm.rubric.models`. It does **not** alter rubric semantics, item-generation logic, score interpretation, psychometric estimation, likelihoods, uncertainty, Rust numerical kernels, persistence, or provider behavior.

Protected `main` previously called `operator.index()` before proving that a caller-controlled value had an exact package-trusted scalar identity. Arbitrary `__index__` providers and caller-defined NumPy integer subclasses could therefore execute caller code while immutable `RubricLevel`, `BlueprintPlan`, or `ItemBlueprint` controls were being normalized.

## Security rationale

The correction uses positive exact-type admission before conversion. Built-in `int` values are accepted directly. Genuine NumPy integer scalar classes represented by NumPy's supported integer type codes are admitted only by class identity and are then converted with `operator.index()`. Booleans, arbitrary integer-like protocol objects, and caller-defined Python or NumPy integer subclasses are rejected before conversion or range comparisons can dispatch caller code.

This maps to CWE-1287 because the vulnerable boundary accepted values as an expected integer type too broadly before a caller-dispatchable normalization step. OWASP ASVS 5.0.0 V2.2.1 requires positive validation against expected values, patterns, ranges, structures, and logical limits; the implementation preserves all existing score, item-count, replicate-index, seed, and unsigned-64 domains while narrowing only type admission.

NIST SP 800-218 SSDF 1.1 remains the current final normative SSDF publication. NIST SP 800-218 Rev. 1 / SSDF 1.2 was released as an Initial Public Draft on December 17, 2025 and is tracked as draft guidance rather than represented as final normative guidance.

## Verification contract

Public-constructor regressions exercise arbitrary `__index__` providers, caller-defined Python integer subclasses, caller-defined NumPy integer subclasses, and genuine NumPy integer scalars. Rejected hostile controls must execute zero caller callbacks. Genuine supported NumPy scalars must normalize to built-in `int` values and retain the established numeric bounds.

The RED commit is retained as test-first lineage only. After implementation changes, CI, security, coverage, package, provenance, and review evidence must be regenerated on the exact current head. Automated review evidence does not replace protected-main independent non-author approval or last-push approval requirements.

## References

MITRE Corporation. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0: V2.2 input validation*. https://owasp.org/www-project-application-security-verification-standard/
