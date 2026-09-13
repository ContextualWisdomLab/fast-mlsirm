# Paired rating-range category-count trust boundary

## Scope

`fast_mlsirm.rating_range` uses Python only to validate and marshal paired ordinal rating inputs before the compiled Rust rating-range kernel runs. Protected `main` admitted any `int`/`np.integer` subclass through `isinstance(...)` and then normalized the public `category_count` control with `int(...)`. A caller-defined integer subclass could therefore execute conversion code before the package established a trusted category-count identity.

The correction admits only an exact built-in Python integer or an exact package-supported genuine NumPy integer scalar class. Supported NumPy integer scalars are normalized to a built-in integer only after exact identity admission. Caller-defined Python and NumPy subclasses are rejected before integer conversion, type hashing/equality, representation, or result-affecting Rust dispatch can execute caller-controlled code.

## Scientific boundary

This change does not alter rating-range evidence arithmetic or interpretation. Paired sample size, observed category endpoints, distinct-category counts, spans, population standard deviations, ratios, endpoint gaps, narrower-support evidence, and central-tendency evidence remain Rust-owned. The existing public `category_count` domain of 2 through 1000 is unchanged, and no threshold is introduced for accepting or rejecting an automated scorer.

## Security and reliability interpretation

MITRE CWE-1287 classifies failure to validate an input's specified type as an input-validation weakness and recommends an accept-known-good strategy. Exact scalar-class admission applies that principle at the Python/Rust marshalling boundary without using caller-controlled conversion or metadata as trust evidence.

NIST SP 800-218, SSDF Version 1.1, remains the final secure-development baseline. NIST SP 800-218 Rev. 1 / SSDF Version 1.2 is tracked separately as an Initial Public Draft, not represented as final guidance. OWASP ASVS 5.0.0 is the current stable ASVS release; its positive input-validation guidance is informative here without implying certification or conformance.

## Verification contract

- a caller-defined Python `int` subclass must fail without executing `__int__` or `__repr__` and without reaching the Rust loader;
- a caller-defined NumPy integer subclass must fail without executing integer conversion, representation, or hostile metaclass hashing/equality and without reaching the Rust loader;
- genuine supported NumPy integer scalars remain accepted and normalize to the same built-in value before dispatch;
- booleans and controls outside `2..=1000` retain package-owned `ValueError` behavior;
- Python computes no paired rating-range statistic.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (Version 4.20).* Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0.* https://owasp.org/www-project-application-security-verification-standard/
