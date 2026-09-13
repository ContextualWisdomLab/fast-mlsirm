# Exploratory model factor-count callback safety

## Scope

The public exploratory-model contract accepts a positive latent-dimension count. This boundary is control marshalling, not psychometric estimation: production likelihood, gradients, optimization, identification, factor retention, loading estimation, rotation, uncertainty, and structural model selection remain unchanged and keep their existing numerical ownership.

Protected-main behavior previously accepted broad `int`/`numpy.integer` subclasses and then invoked `int(...)`. In Python, a caller-defined subclass can override integer conversion, so nominal type acceptance could execute caller-controlled code before the package had established a trusted factor-count value.

The corrected boundary uses positive validation. Exact built-in Python integers are accepted directly; exact genuine NumPy integer scalar classes are normalized with `int(...)`; integer subclasses and other objects are rejected before conversion or representation callbacks can execute. The existing positive-factor requirement is preserved. Multidimensional exploratory estimation remains explicitly unsupported pending the separately governed identified estimator work in #633.

## Security and reliability interpretation

This is a specified-input-type validation defect rather than a new estimator or model formula. MITRE CWE-1287 recommends validating the expected input type and using an accept-known-good strategy. The fix therefore narrows trusted control identities instead of trying to detect particular hostile callbacks after execution.

The repository continues to treat NIST SP 800-218, SSDF Version 1.1, as the final secure-development baseline. NIST SP 800-218 Rev. 1 / SSDF Version 1.2 is an Initial Public Draft as of this doctoring record and is tracked as draft guidance rather than represented as final. OWASP ASVS 5.0.0 is the current stable ASVS release; its input-validation guidance is informative here because this package boundary validates a caller-controlled control value, but this change does not claim ASVS conformance or certification.

## Verification contract

Regression evidence must prove both sides of the boundary:

- genuine built-in and NumPy integer scalar factor counts remain accepted;
- Python and NumPy integer subclasses are rejected before hostile `__int__` or `__repr__` callbacks execute;
- direct `exploratory(...)` and `_resolve_model(...)` behavior remain consistent;
- one-factor exploratory resolution is unchanged;
- multidimensional exploratory requests continue to fail explicitly rather than silently changing estimator behavior.

## References

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software, 48*(6), 1–29. https://doi.org/10.18637/jss.v048.i06

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (Version 4.20).* Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (n.d.). *OWASP Application Security Verification Standard 5.0.0.* https://owasp.org/www-project-application-security-verification-standard/
