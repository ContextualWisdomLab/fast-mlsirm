# Enterprise evidence integer callback trust boundary

## Decision

Enterprise source character counts and evidence offsets are validation metadata, not psychometric arithmetic. They remain in the Python contract layer, but the package now establishes exact scalar identity before normalization. Exact built-in `int` values and genuine package-supported NumPy integer scalar identities are accepted; booleans, arbitrary `__index__` providers, caller-defined Python integer subclasses, and caller-defined NumPy integer subclasses fail closed before caller conversion, representation, comparison, equality, or hashing callbacks can participate.

The change preserves existing nonnegative bounds, nonempty evidence-span semantics, source-text-free content addressing, and package-owned `AssessmentSpecError` codes. It does not change scoring, calibration, ranking, utility, causal, sentiment, likelihood, estimator, uncertainty, Rust numerical kernels, provider behavior, persistence, or downstream Psychometrics Commons ownership.

## Security basis

OWASP ASVS 5.0.0 V2.2.1 permits positive allowlist validation against expected values, patterns, and ranges or comparison with an expected structure and logical limits under predefined rules. V2.2.2 requires input validation at a trusted service layer and states that client-side validation must not be relied upon as a security control. Exact scalar identity is the allowlist boundary here; Python's executable integer protocol is not treated as authority.

The correction also follows the NIST Secure Software Development Framework's defect-prevention and root-cause-remediation intent. NIST SP 800-218 (SSDF 1.1) remains final; SP 800-218 Rev. 1 (SSDF 1.2) is newer but remains an Initial Public Draft and is not represented as final guidance.

The defect is consistent with CWE-1287 (Improper Validation of Specified Type of Input): integer-shaped values were previously admitted through a caller-dispatchable protocol before trusted type validation.

## Regression evidence

`tests/test_scoring_enterprise_issue_integer_callback_boundary.py` exercises all governed integer controls through the public enterprise contract surface. Adversarial fixtures use an arbitrary `__index__` provider, a Python `int` subclass, and a NumPy integer subclass and require zero recorded caller callbacks. A compatibility regression requires genuine signed and unsigned NumPy integer scalars to normalize to exact built-in integers.

## References

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

MITRE. (n.d.). *CWE-1287: Improper validation of specified type of input*. https://cwe.mitre.org/data/definitions/1287.html
