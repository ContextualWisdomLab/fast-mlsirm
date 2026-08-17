# Doctoring: remaining equating control safety

## Decision

Public observed-score equating adapters validate semantic controls before compiled-core discovery. The Python layer accepts only exact built-in primitives and genuine supported NumPy scalar identities for scalar controls, rejects caller-defined subclasses and arbitrary conversion providers without invoking their callbacks, and then marshals normalized values to Rust. Result-affecting circle-arc geometry, nominal-weights moments, composite-linking weights, and all other equating arithmetic remain Rust-owned.

The bounded correction applies to `circle_arc_equate`, `circle_arc_middle_anchor`, `nominal_weights_mean_equate`, and `composite_linking`. Circle-arc method validation mirrors the Rust parser exactly (`1`, `arc1`, `circlearc1`, `2`, `arc2`, `circlearc2`, case-insensitive) rather than using punctuation normalization that would widen the native vocabulary. Composite-linking `p` is normalized as a finite trusted real and is rejected below 1 before native discovery, matching the Rust domain.

## Threat model and verification

The defect class is type-boundary validation: a nominally scalar/configuration argument could be a Python object whose `__str__`, `__int__`, `__index__`, `__float__`, comparison, hashing, or container protocol executes caller-owned code. Validation after native discovery also made rejected configuration depend on extension-loader availability. The regression suite therefore uses hostile protocol providers and Python/container subclasses whose callbacks raise if executed, replaces native discovery with a sentinel, and asserts both zero callback execution and zero core discovery for rejected controls. It separately proves genuine NumPy integer/float scalar compatibility and built-in normalization at the Rust-shaped call boundary.

This maps directly to CWE-1287, which recommends accept-known-good input validation for values expected to have a specified type. OWASP ASVS 5.0.0 is the current stable ASVS baseline and is used here as secure-input-validation guidance. NIST SP 800-218 SSDF 1.1 remains the final SSDF baseline; SP 800-218 Rev. 1 / SSDF 1.2 is an Initial Public Draft and is treated as draft, not normative final authority.

## Scientific ownership

No estimand, objective, likelihood, moment equation, transformation, circle geometry, or Holland–Strawderman weight computation moved into Python. The numerical implementation remains in `crates/mlsirm-core/src/equating.rs`. Python performs only inert validation, normalization, array marshalling, dispatch, and result wrapping. Existing domain references for the implemented mathematics remain authoritative, including Livingston and Kim (2008), Babcock et al. (2012), and Albano (2016).

## References

Albano, A. D. (2016). equate: An R package for observed-score linking and equating. *Journal of Statistical Software, 74*(8), 1–36. https://doi.org/10.18637/jss.v074.i08

Babcock, B., Albano, A., & Raymond, M. (2012). Nominal weights mean equating: A method for very small samples. *Educational and Psychological Measurement, 72*(4), 608–628. https://doi.org/10.1177/0013164411428609

Livingston, S. A., & Kim, S. (2008). *Small-sample equating by the circle-arc method* (Research Report No. RR-08-39). ETS. https://doi.org/10.1002/j.2333-8504.2008.tb02135.x

MITRE. (2026). *CWE-1287: Improper validation of specified type of input* (CWE 4.20). https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure Software Development Framework (SSDF) Version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure Software Development Framework (SSDF) Version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/
