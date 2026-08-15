# Parallel analysis public control and workspace bounds

## Standards and literature

Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179–185. https://doi.org/10.1007/BF02289447

Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement, 55*(3), 377–393. https://doi.org/10.1177/0013164495055003002

MITRE Corporation. (2026). *CWE-1287: Improper validation of specified type of input (CWE 4.20).* Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities (NIST Special Publication 800-218).* U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-218

Open Worldwide Application Security Project. (2025). *OWASP Application Security Verification Standard 5.0.0: V2.2 input validation.* OWASP Foundation. https://cornucopia.owasp.org/taxonomy/asvs-5.0/02-validation-and-business-logic/02-input-validation

### Standards watch — non-normative draft

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities (NIST Special Publication 800-218 Rev. 1, Initial Public Draft).* National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd

The 2025 SSDF 1.2 document remains an initial public draft, so the final SSDF 1.1 publication remains the normative NIST reference for this change. The draft is tracked only for standards-watch continuity.

## Product application

Public `parallel_analysis` controls (`n_iterations`, `centile`, `seed`) use positive validation against exact trusted scalar identities before any caller-controlled integer conversion or compiled-core discovery. Exact built-in Python integers and genuine supported NumPy integer scalar classes are admitted; booleans, arbitrary integer-like objects, and caller-defined Python/NumPy integer subclasses fail closed without invoking their conversion or representation hooks. This applies CWE-1287's expected-type validation principle and ASVS 5.0.0 V2.2's positive-validation/trusted-layer requirements at the Python-to-native trust boundary.

The controls retain their established domains: positive `n_iterations`, `centile` in `0..=99`, and `seed` in the Rust `u64` domain. Iteration counts that would request more than the package's 128 MiB random-eigenvalue workspace ceiling are rejected before compiled dispatch, with a corresponding Rust allocation guard remaining defense in depth.

These trust-boundary changes do not alter Horn/Glorfeld factor-retention mathematics, the deterministic random stream, eigenvalue adjustment, centile semantics, or retention scanning. Result-affecting factor-retention arithmetic remains Rust-owned; Python validates and marshals only.

## Verification

- `tests/test_parallel_analysis_control_bounds.py` proves hostile Python and NumPy integer subclasses fail before native discovery and before conversion/representation callbacks.
- The same regression surface preserves genuine NumPy integer controls and existing range/workspace rejection.
