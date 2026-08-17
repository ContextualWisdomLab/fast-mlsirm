# CDM native-control validation trust boundary

## Status

Implemented on the issue #902 source branch until protected-main integration. This is a bounded public-input validation correction for the cognitive-diagnosis Python adapters; it does not change DINA, DINO, G-DINA, higher-order, sequential, PVAF, or Wald psychometric arithmetic in the Rust core.

## Problem

The CDM-family adapters previously discovered the compiled native core before validating caller-facing controls. Their shared stopping-control validator accepted broad Python/NumPy numeric classes and then normalized them through `int(...)` / `float(...)`; the DINA/DINO adapters similarly normalized `model` with `str(...)` at native dispatch. Python special-method dispatch means those conversions are executable behavior for caller-defined subclasses and protocol providers, rather than passive type checks. A rejected configuration could therefore execute caller-owned conversion or representation callbacks, or touch the native-loader boundary, before the package had established that the control value belonged to the documented input domain.

This is treated as a specified-type validation boundary consistent with CWE-1287. OWASP ASVS 5.0.0 V2.2.1 likewise requires validation against expected structure and logical limits, while V2.2.2 places such enforcement in a trusted service layer. NIST SP 800-218 SSDF Version 1.1 is used as secure-development-process context; later draft revisions are not represented here as final standards. These references are engineering guidance and taxonomy, not certification or severity claims.

## Decision

For the affected public CDM adapters:

- validate the response/Q-matrix structure and public controls before `_core_module()` discovery;
- admit `max_iter` only when its runtime type is exactly built-in `int` or one of the explicitly supported concrete NumPy integer scalar types, then enforce the repository iteration cap;
- admit `tol` only when its runtime type is exactly built-in `int`/`float` or one of the explicitly supported concrete NumPy integer/floating scalar types, then require a finite strictly positive value;
- reject booleans, Python/NumPy subclasses, and arbitrary conversion-protocol providers before calling their conversion or comparison hooks;
- admit DINA/DINO selectors only as exact built-in strings in the existing vocabulary `{"dina", "dino"}` and pass the trusted string directly to PyO3; and
- keep all result-affecting likelihood, EM, model-selection, classification, convergence, uncertainty, PVAF, Wald, higher-order, and sequential arithmetic in Rust unchanged.

## Evidence contract

The fail-first commit `9409e5dd3158fb9b57a999645573b44e9db4a0f2` exercises the public adapters with Python integer/float subclasses, NumPy scalar subclasses, arbitrary numeric protocol providers, and a string subclass. Each hostile value increments a counter from any conversion/comparison hook, while native-core discovery is replaced by a sentinel that must remain unreachable for invalid input.

GREEN requires all of the following:

- hostile `max_iter`, `tol`, and `model` controls raise stable package-owned `ValueError`s with callback counts remaining zero;
- malformed public inputs are rejected before native discovery;
- booleans, non-finite tolerances, out-of-range iteration caps, and unknown model strings remain rejected;
- genuine supported concrete NumPy scalar controls and exact `"dina"`/`"dino"` strings still reach the established native-dispatch boundary;
- valid inputs retain the existing compiled-core-required `RuntimeError` when the extension is unavailable; and
- package Python/Rust/PyO3, coverage, security, package, fuzz, provenance, and required-review gates remain authoritative integration evidence on the exact PR head.

The implementation commit `66cb895f16efc10d3a1f6afadbe6934b665085df` changes validation and marshalling order only. Exact-head CI remains the integration authority; predecessor-head results do not transfer after a source update.

## Scope limitations

This correction does not redesign CDM formulas, estimators, convergence criteria, Q-matrix identifiability theory, missing-data assumptions, or structural model selection. It does not introduce a Python numerical fallback or transfer arithmetic ownership away from Rust. Other caller-control boundaries outside the affected CDM adapters require their own bounded regressions rather than being inferred safe from this change.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (CWE version 4.20)*. Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure Software Development Framework (SSDF) Version 1.1: Recommendations for mitigating the risk of software vulnerabilities from software development practices (NIST SP 800-218)*. https://doi.org/10.6028/NIST.SP.800-218

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

Python Software Foundation. (2026). *Data model*. Python 3.14 documentation. https://docs.python.org/3.14/reference/datamodel.html

NumPy Developers. (2026). *Scalars*. NumPy reference. https://numpy.org/doc/stable/reference/arrays.scalars.html
