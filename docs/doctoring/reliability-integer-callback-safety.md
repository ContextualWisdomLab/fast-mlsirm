# Reliability integer-control callback trust boundary

## Decision

The public Finn, Guttman lambda, and Feldt-alpha-CI surfaces treat `s_levels`, `n_sample_splits`, `seed`, `n_persons`, and `n_items` as configuration data rather than executable conversion providers. Exact built-in `int` identities and genuine supported NumPy integer scalar identities are admitted, normalized to Python `int`, range-checked, and only then allowed to reach ratings/response materialization or compiled-core discovery.

This is validation and marshalling only. Finn (1970) mean-square arithmetic, Guttman (1945) lambda/split-half arithmetic, and Feldt (1965) exact-F interval mapping remain owned by the Rust implementation.

## Security basis

The control boundary follows an allow-known-good strategy consistent with CWE-1287, *Improper Validation of Specified Type of Input*. OWASP ASVS 5.0.0 is used as the stable application-security validation baseline. NIST SP 800-218 SSDF 1.1 remains the final secure-development baseline; the newer SP 800-218 Rev. 1 / SSDF 1.2 material is an Initial Public Draft and is treated as informative rather than final authority.

## Psychometric basis

Finn's coefficient measures departure of categorical ratings from a uniform discrete scale (Finn, 1970, as implemented by Gamer et al., 2019). Guttman lambdas bound internal consistency from the item correlation matrix (Guttman, 1945, as implemented by Revelle, 2025). Feldt's exact-F interval maps coefficient alpha sampling error through an F pivot (Feldt, 1965, as implemented by Revelle, 2025). None of those procedures requires executing caller-defined `__int__`, `__index__`, or comparison callbacks to accept a scale length, split budget, seed, or sample-size control.

## Regression evidence

`tests/test_reliability_integer_callback_boundary.py` requires hostile Python/NumPy integer subclasses, arbitrary `__int__`/`__index__` providers, and booleans to fail before native discovery or array materialization with zero caller callbacks. It also proves genuine NumPy integer scalars normalize to exact Python ints and that Finn `s_levels=5` and `s_levels=np.int64(5)` dispatch identically.

## References

Finn, R. H. (1970). A note on estimating the reliability of categorical data. *Educational and Psychological Measurement, 30*(1), 71-76. https://doi.org/10.1177/001316447003000106 (as cited in Gamer et al., 2019; not read)

Guttman, L. (1945). A basis for analyzing test-retest reliability. *Psychometrika, 10*(4), 255-282. https://doi.org/10.1007/BF02288892 (as cited in Revelle, 2025)

Feldt, L. S. (1965). The approximate sampling distribution of Kuder-Richardson reliability coefficient twenty. *Psychometrika, 30*(3), 357-370. https://doi.org/10.1007/BF02289511 (as cited in Revelle, 2025)

Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr: Various coefficients of interrater reliability and agreement* [R package]. https://CRAN.R-project.org/package=irr

Revelle, W. (2025). *psych: Procedures for psychological, psychometric, and personality research* (Version 2.6.5) [R package]. https://CRAN.R-project.org/package=psych

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

MITRE. (n.d.). *CWE-1287: Improper validation of specified type of input*. https://cwe.mitre.org/data/definitions/1287.html
