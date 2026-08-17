# ICC semantic-control callback trust boundary

## Decision

The public ICC surface treats `model`, `type`, `unit`, `r0`, and `conf_level` as configuration data rather than executable conversion providers. Exact built-in strings are allowlisted for the Rust vocabulary. Numeric controls accept exact built-in `int`/`float` identities and genuine supported NumPy integer/floating scalar identities, normalize only those trusted values, reject non-finite and out-of-range values locally, and do so before native-core discovery or ratings materialization.

This is validation and marshalling only. ICC ANOVA arithmetic, F/Satterthwaite calculations, p-values, confidence limits, listwise missing-row handling, and result construction remain owned by the Rust implementation.

## Security basis

The control boundary follows an allow-known-good strategy consistent with CWE-1287, *Improper Validation of Specified Type of Input*. OWASP ASVS 5.0.0 is used as the stable application-security validation baseline. NIST SP 800-218 SSDF 1.1 remains the final secure-development baseline; the newer SP 800-218 Rev. 1 / SSDF 1.2 material is an Initial Public Draft and is treated as informative rather than final authority.

## Regression evidence

`tests/test_reliability_icc_control_callbacks.py` requires invalid vocabulary values, arbitrary protocol providers, Python/NumPy scalar subclasses, booleans, complex values, non-finite values, and Rust-range violations to fail before ratings access or native dispatch while proving zero caller conversion/comparison/hash callbacks. It also proves genuine NumPy real scalars normalize to exact Python floats. `tests/test_reliability_icc_control_extremes.py` covers exact built-in integers too large to represent as finite `float64`, requiring a package-owned `ValueError` before ratings or core access.

## References

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

MITRE. (n.d.). *CWE-1287: Improper validation of specified type of input*. https://cwe.mitre.org/data/definitions/1287.html
