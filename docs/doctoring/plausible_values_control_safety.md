# Plausible-value control trust-boundary hardening

## Scope

Issue #914 changes only the Python validation and marshalling boundary for `fast_mlsirm.serving.plausible_values`. The Rust core continues to own posterior reduction, plausible-value sampling, device selection/fallback, and all psychometric/statistical arithmetic. No estimator, likelihood, quadrature rule, uncertainty interpretation, or result schema changes in this slice.

Protected `main` previously discovered the compiled core before validating three caller-controlled request parameters and then used generic Python coercion for them. In particular, `n_draws` and `seed` could reach `int(...)`, while `device` could reach `str(...)`. Python subclasses and arbitrary conversion providers can attach executable callbacks to those coercions. The public boundary also did not explicitly enforce the Rust/PyO3 `u64` domain for `seed`.

## Security contract

The corrected boundary establishes the following order before `_core_module()` is called:

1. validate the serving bundle;
2. accept `n_draws` and `seed` only when their runtime type is the exact built-in `int` or one of the exact supported concrete NumPy integer scalar identities;
3. normalize an admitted NumPy scalar once to a built-in integer;
4. enforce `1 <= n_draws <= MAX_DRAWS` and `0 <= seed <= 2**64 - 1`;
5. accept `device` only when its runtime type is the exact built-in `str` and its value is one of `cpu`, `gpu`, or `auto`;
6. validate and marshal response/prior data; and only then
7. discover and call the Rust core.

Trusted NumPy scalar admission deliberately uses `is` identity comparisons instead of set membership or equality. Hash- or equality-based admission would permit a caller-controlled scalar metaclass to execute `__hash__` or `__eq__` during the trust decision. The regression suite therefore covers conversion callbacks and scalar-metaclass callbacks independently.

This is positive/allow-list validation at the trusted service boundary, consistent with CWE-1287 and OWASP ASVS v5.0.0-2.2.1/v5.0.0-2.2.2. OWASP identifies ASVS 5.0.0 as the latest stable release as rechecked on 2026-08-16. MITRE CWE 4.20 identifies CWE-1287 as the specific base weakness for input whose specified type is not correctly validated and recommends an accept-known-good strategy. NIST SP 800-218 SSDF 1.1 remains the final normative SSDF baseline; SP 800-218 Rev. 1 / SSDF 1.2 is recorded only as an Initial Public Draft standards-watch item as of the same recheck.

## Test-first evidence

- `acfb554184208cce2afd5ed20c8f8b916c98dcfd` specifies the initial RED contract for hostile integer/string subclasses, unsupported devices, Rust-`u64` seed bounds, pre-core rejection, and supported NumPy scalar compatibility.
- `7fc499fbbde4d2518a5afa93c37e3d9235955da2` supplies the first bounded GREEN by validating and normalizing controls before core discovery.
- `8531dbdb2b0f53497bcf0f94a97873a888bbe668` adds a second RED demonstrating that hashed trusted-type membership still executes a caller-controlled scalar metaclass callback.
- `bbc640a97c556b0241e3bc33e0b24f3904a50612` replaces hashed/equality-based type admission with identity-only comparisons.
- The public suite also proves pre-core rejection for `bool` / `np.bool_`, non-integer floats, `n_draws` domain `0` and `MAX_DRAWS + 1`, `__index__` providers, and that a valid request discovers the compiled core exactly once at dispatch.

Hosted exact-head CI, security, package/provenance, coverage/OpenCode, formal-review, and protected-base evidence is authoritative over these remembered commit identities and must be refetched before any lifecycle or integration decision.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). U.S. Department of Commerce. https://csrc.nist.gov/pubs/sp/800/218/r1/ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://github.com/OWASP/ASVS/tree/v5.0.0_release
