# G-theory NumPy scalar trust boundary

## Scope

`fast_mlsirm.gtheory` uses Python only to validate and marshal public controls before the compiled Rust G-theory kernels run. Protected-main validation distinguished supposedly trusted NumPy numeric scalars by checking both `isinstance(...)` and whether `type(value).__module__` started with `numpy`. Class metadata is caller-controlled for user-defined subclasses, so a subclass of a NumPy scalar could spoof that string and reach `int(...)` or `float(...)` with an overridden conversion method.

The correction replaces metadata-based trust with exact package-supported NumPy scalar type identities. Built-in Python controls retain their existing exact-type handling. Genuine NumPy integer and floating scalar classes remain accepted. User-defined subclasses are rejected before conversion or representation callbacks can execute, including subclasses that claim a `numpy` module name.

## Scientific boundary

This change does not alter generalizability-theory arithmetic. ANOVA sums of squares and mean squares, variance-component estimation, the repository's documented negative-component clamping policy for D-study quantities, generalizability/dependability coefficients, Phi(lambda), denominator handling, and all result-affecting numerical work remain Rust-owned. The worked Huebner and Lucht (2019) G-study/D-study reference cases remain the scientific oracle for the existing formulas; callback-safety tests only exercise Python control marshalling.

## Security and reliability interpretation

The defect is improper validation of a specified input type: caller-controlled metadata was used as evidence that conversion behavior belonged to NumPy. MITRE CWE-1287 recommends an accept-known-good validation strategy for expected input types. Exact scalar-class allowlists make that trust decision independent of caller-modifiable module metadata.

NIST SP 800-218, SSDF Version 1.1, remains the final secure-development baseline recorded here. NIST SP 800-218 Rev. 1 / SSDF Version 1.2 is an Initial Public Draft and is tracked as draft guidance rather than represented as final. OWASP ASVS 5.0.0 is the current stable ASVS release; its positive input-validation guidance is informative to this boundary without implying package certification or ASVS conformance.

## Verification contract

- a caller-defined `np.integer` subclass with spoofed NumPy module metadata must fail without executing `__int__` or `__repr__`;
- a caller-defined `np.floating` subclass with spoofed NumPy module metadata must fail without executing `__float__` or `__repr__`;
- accepted Python and genuine NumPy integer D-study sizes preserve their ordinary marshalled integer values;
- accepted Python and genuine NumPy real Phi(lambda) cuts preserve their ordinary finite values;
- booleans, non-positive D-study sizes, non-real controls, and non-finite cuts retain the existing stable validation errors;
- rejected controls never reach result-affecting Rust G-theory calls.

## References

Huebner, A., & Lucht, M. (2019). Generalizability theory in R. *Practical Assessment, Research, and Evaluation, 24*, Article 5. https://doi.org/10.7275/5065-gc10

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (Version 4.20).* Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (n.d.). *OWASP Application Security Verification Standard 5.0.0.* https://owasp.org/www-project-application-security-verification-standard/
