# Hofstee scalar-control trust boundary

## Scope

This note documents the input-validation boundary added for the four public Hofstee standard-setting controls (`min_cut`, `max_cut`, `min_fail`, and `max_fail`). It does not alter the Hofstee ogive, intersection, directed-rounding fallback, or any other psychometric/statistical arithmetic, which remains owned by the Rust core.

The Python adapter establishes a package-trusted scalar identity before compiled-core discovery. It accepts exact built-in `int`/`float` values and genuine supported NumPy integer/floating scalar identities, rejects booleans, scalar subclasses, and arbitrary conversion-protocol providers before caller callbacks, normalizes accepted values once to built-in `float`, then enforces finite `[0, 100]` domains and ordered bound pairs.

## Security rationale

The boundary follows an allowlisted validation strategy at the trusted layer rather than invoking caller-controlled coercion to discover whether a value is admissible. This reduces callback/re-entrancy behavior during validation and keeps malformed control inputs from reaching native dispatch.

- CWE-1287 identifies insufficient specified-type validation as a weakness and recommends validating against known-good expected type/range contracts.
- OWASP ASVS 5.0.0 is used as the current stable application-security verification baseline; its validation guidance places input validation at a trusted service layer.
- NIST SP 800-218 SSDF 1.1 remains the final secure-development baseline used here. The later SP 800-218 Rev. 1 / SSDF 1.2 publication is still draft material and is tracked as non-normative until finalized.

## Verification contract

Regression tests require rejected Python/NumPy scalar subclasses and arbitrary protocol providers to execute zero conversion, representation, comparison, equality, or hashing callbacks and to cause zero Rust-core discovery. Separate cases cover booleans, non-finite values, percentage-range violations, built-in-integer overflow during trusted normalization, inverted cut/fail bounds, genuine NumPy scalar compatibility, and exact built-in-float marshalling at the Rust boundary.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218). U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218 Rev. 1, Initial Public Draft). U.S. Department of Commerce.

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/
