# Parallel-analysis control trust boundary

## Decision

`parallel_analysis()` treats `n_iterations`, `centile`, and `seed` as inert configuration data. The Python boundary admits only exact built-in `int` values and exact package-supported concrete NumPy integer scalar identities, validates their established domains, and does so before native-core discovery. Caller-defined integer subclasses and arbitrary numeric-conversion providers are rejected without invoking conversion or representation hooks.

The validation change is intentionally non-numerical. Horn/Glorfeld factor-retention arithmetic, the deterministic random stream, correlation/eigenvalue computation, random-eigenvalue adjustment, retention scanning, and all result-affecting production mathematics remain Rust-owned.

## Security basis

This is an accept-known-good type-validation boundary consistent with CWE-1287, which recommends validating the expected type and rejecting inputs outside the accepted specification. OWASP ASVS 5.0.0 remains the current stable ASVS release and is used as the application-security validation baseline. NIST SP 800-218 SSDF 1.1 remains final; SP 800-218 Rev. 1 / SSDF 1.2 is still an Initial Public Draft and is tracked as informative draft material rather than represented as final authority.

## Regression evidence

`tests/test_parallel_analysis_control_bounds.py` requires rejected scalar values, `np.bool_`, Python/NumPy integer subclasses, arbitrary conversion providers, out-of-domain controls, and oversized random-benchmark workspaces to fail before native-core discovery. It also proves the allowlisted concrete NumPy integer scalar identities, including distinct `longlong`/`ulonglong` types, normalize to exact Python integers at the Rust boundary while preserving the existing iteration, centile, seed, and random-workspace limits.

## References

Horn, J. L. (1965). A rationale and a test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179-185. https://doi.org/10.1007/BF02289447

Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement, 55*(3), 377-393. https://doi.org/10.1177/0013164495055003002

Dinno, A. (2018). *paran: Horn's test of principal components/factors* (Version 1.5.6) [R package]. https://CRAN.R-project.org/package=paran

MITRE. (2026). *CWE-1287: Improper validation of specified type of input* (CWE 4.20). https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/
