# IRT linking method-control validation

## Decision

The public Python `irt_link()` boundary validates `method` before loading or calling the compiled Rust core. The accepted vocabulary is exactly the existing Rust `LinkMethod::parse` contract: mean/mean (`mean_mean`, `mean-mean`, `meanmean`, `mm`), mean/sigma (`mean_sigma`, `mean-sigma`, `meansigma`, `ms`), Haebara (`haebara`, `hb`), and Stocking-Lord (`stocking_lord`, `stocking-lord`, `stockinglord`, `sl`), with case-insensitive matching.

Only an exact built-in Python `str` is accepted. Arbitrary objects and `str` subclasses are rejected before normalization so caller-controlled `__str__`, `__repr__`, `lower`, or other overridden text callbacks cannot become part of package validation. A trusted accepted string is then passed unchanged to Rust, which remains the authoritative numerical implementation and parser.

## Defect and threat model

Protected main previously used `str(method)` immediately before the PyO3 call and again while constructing `IrtLinkResult`. That conversion is executable behavior for an arbitrary object and can raise caller-defined exceptions, perform side effects, or consume resources before package-owned validation. A `str` subclass can additionally replace string-normalization methods. This is a marshalling and trust-boundary defect, not a defect in the Haebara, Stocking-Lord, mean/mean, or mean/sigma formulas.

The fix applies allowlisted type and lexical validation at the earliest public-control boundary. This follows the specified-type validation principle in CWE-1287 and prevents caller exceptions from escaping as the package's control-flow contract (CWE-248). OWASP ASVS 5.0.0 likewise treats input validation as an explicit application-security verification area. Final NIST SP 800-218 SSDF 1.1 remains the released SSDF authority used by this repository; NIST SP 800-218 Rev. 1 / SSDF 1.2 is tracked separately as an initial public draft published December 17, 2025 rather than represented as final guidance.

## Test-first evidence

- RED `84833cc6d97c02f2840738d972ea459a9686b484` requires hostile objects, hostile `str` subclasses, and unsupported built-in method names to fail before the compiled-core loader, without executing caller representation or normalization callbacks. It also pins successful preservation of the Rust-supported `SL` alias.
- GREEN `d937b65395253173d1e1b3e8a8c4276d090b0b7b` adds the exact-string allowlist validator, calls it before native-loader access, removes both `str(method)` calls, and passes the trusted method identity unchanged to the Rust boundary and result object.

The regression is deliberately Python-only because it validates Python marshalling behavior. All result-affecting linking arithmetic, moment transformations, characteristic-curve objectives, Nelder-Mead optimization, convergence diagnostics, and coefficients remain in `mlsirm-core` Rust.

## Scientific boundary

No likelihood, linking equation, weighting scheme, quadrature rule, optimizer criterion, convergence tolerance, parameter transformation, or score interpretation changes. The existing Rust parser and scientific implementation remain authoritative. The method-validation allowlist was derived directly from the live Rust `LinkMethod::parse` vocabulary rather than inventing a second scientific method taxonomy.

## Verification and rollback

Verification should run the focused callback-boundary tests and existing linking coverage first, then changed-file lint/docstring/branch-coverage checks, the Rust workspace tests, package checks, security/static analysis, and the exact-head hosted review gates. A rollback removes the Python validation change and its tests/docs only; it does not alter or migrate numerical code.

## References (APA 7th ed.)

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities (NIST SP 800-218 Rev. 1, Initial Public Draft).* National Institute of Standards and Technology. https://csrc.nist.gov/pubs/sp/800/218/r1/ipd

Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking: Methods and practices* (3rd ed.). Springer. https://doi.org/10.1007/978-1-4939-0317-7

MITRE. (n.d.). *CWE-248: Uncaught exception.* Retrieved August 14, 2026, from https://cwe.mitre.org/data/definitions/248.html

MITRE. (n.d.). *CWE-1287: Improper validation of specified type of input.* Retrieved August 14, 2026, from https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities (NIST SP 800-218).* https://doi.org/10.6028/NIST.SP.800-218

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0.* https://owasp.org/www-project-application-security-verification-standard/
