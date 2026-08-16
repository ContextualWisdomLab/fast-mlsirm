# Essay integer callback trust boundary

## Decision

Essay prompt limits, submission counts, and evidence offsets are validation metadata, not psychometric arithmetic. They therefore remain in the Python contract layer, but their scalar identity must be established before any conversion protocol is dispatched. The public essay factories accept exact built-in `int` values and genuine package-supported NumPy integer scalar identities. Booleans, arbitrary `__index__` providers, caller-defined Python integer subclasses, and caller-defined NumPy integer subclasses fail closed before caller conversion, representation, comparison, equality, or hashing callbacks can participate.

This preserves the existing nonnegative ranges and package-owned `AssessmentSpecError` codes. It does not change scoring, calibration, estimation, uncertainty, likelihoods, Rust kernels, provider behavior, persistence, or downstream Psychometrics Commons ownership.

## Security basis

OWASP ASVS 5.0.0 V2.2.1 requires positive validation of input against expected values, structures, and logical limits, while V2.2.2 places security validation at a trusted service layer. The exact-type allowlist used here applies that principle before coercion rather than treating Python's executable integer protocol as authority.

The correction also follows the NIST Secure Software Development Framework's defect-prevention and root-cause-remediation intent. NIST SP 800-218 (SSDF 1.1) remains the final normative publication; SP 800-218 Rev. 1 (SSDF 1.2) is an Initial Public Draft and is tracked here as newer draft guidance rather than represented as final.

The defect is consistent with CWE-1287 (Improper Validation of Specified Type of Input): an integer-shaped value was previously accepted through a caller-dispatchable protocol before the package established a trusted scalar type.

## Regression evidence

`tests/test_scoring_essay_integer_callback_boundary.py` exercises every governed essay integer field through the public factories. The adversarial fixtures use an arbitrary `__index__` provider, a Python `int` subclass, and a NumPy integer subclass and require zero recorded caller callbacks. A companion compatibility regression requires genuine signed and unsigned NumPy integer scalars to normalize to exact built-in integers.

## References

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

MITRE. (n.d.). *CWE-1287: Improper validation of specified type of input*. https://cwe.mitre.org/data/definitions/1287.html
