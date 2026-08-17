# Many-facet public control trust boundary

## Scope

This note documents the input-validation boundary for the public `fit_facets(...)` controls `n_cat`, `q_theta`, `max_iter`, and `tol`. It does not alter the many-facet response model, marginal-ML EM, Gauss-Hermite quadrature, connectedness diagnostics, likelihood, optimization, parameter estimates, or identification constraints. Those result-affecting psychometric/statistical operations remain owned by the Rust core.

The Python adapter now establishes package-trusted scalar identity and control domains before compiled-core discovery. Exact built-in integers/reals and genuine supported NumPy integer/floating scalar identities are normalized once to built-in `int`/`float`; booleans, Python/NumPy scalar subclasses, arbitrary conversion-protocol providers, and other caller-defined scalar objects are rejected before their conversion, hashing, equality, ordering, or NumPy ufunc hooks can run.

## Validation contract

- `n_cat` is optional; when supplied it is an exact trusted integer in `2..MAX_POLYTOMOUS_CATEGORIES`.
- `q_theta` is an exact trusted integer and remains restricted to the existing quadrature grid sizes `7`, `11`, `15`, `21`, `31`, or `41`.
- `max_iter` is an exact trusted integer in `1..MAX_MAX_ITER`.
- `tol` is an exact trusted real scalar, normalized to built-in `float`, finite, and strictly positive.
- Response-array shape, finiteness, observed-category integrity, item/rater observability, and inferred-category checks complete before Rust capability discovery.

These are API trust and resource-domain checks only. They are not new scientific recommendations about category counts, quadrature accuracy, iteration budgets, or convergence tolerances.

## Verification contract

Regression tests require rejected Python/NumPy scalar subclasses and arbitrary numeric-protocol providers to execute zero caller callbacks and cause zero Rust-core discovery. Separate cases cover booleans, invalid domains, non-finite/overflowing tolerances, malformed response shape, genuine NumPy scalar compatibility, and exact built-in scalar marshalling at the Rust boundary. Hosted exact-head CI remains responsible for the compiled Rust/PyO3 oracle and package-install evidence.

## Security rationale

The boundary follows allowlisted trusted-type validation rather than invoking caller-controlled coercion to discover whether a value is admissible. This implements the repository's fail-closed public-input convention and reduces re-entrancy during native capability establishment.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218). U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218 Rev. 1, Initial Public Draft). U.S. Department of Commerce.

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/
