# ATA integer callback safety

## Decision

`assemble_to_target()` treats public integer-valued controls as a trust boundary. The package admits exact built-in `int` values and a finite allowlist of genuine NumPy integer scalar classes, then normalizes only those admitted values. Boolean values, caller-defined Python/NumPy integer subclasses, and arbitrary integer-like objects fail before caller conversion callbacks and before item-information work.

This is validation and marshalling only. Automated test assembly information, candidate gain, tie-breaking, content feasibility, exposure behavior, and all other result-affecting numerical logic remain unchanged.

## Rationale

Broad subclass checks such as `isinstance(value, (int, np.integer))` do not establish that subsequent `int(value)` normalization is package-controlled: a caller-defined subclass can override conversion behavior. Exact scalar identity admission makes the accepted type boundary explicit and testable while retaining genuine NumPy interoperability. Range and bank-membership checks still execute after trusted normalization, so type, quantity, and index validity remain separate contracts.

The boundary is consistent with CWE-1287's requirement to validate the expected input type and with OWASP ASVS 5.0.0 input-validation guidance. NIST SP 800-218 SSDF 1.1 remains the current final SSDF authority; SP 800-218 Rev. 1 / SSDF 1.2 is an Initial Public Draft and is treated as draft guidance rather than a final replacement.

## Verification

Focused regressions exercise hostile Python and NumPy integer subclasses at scalar and container controls. Rejected values must produce package-owned `ValueError` results, execute zero hostile conversion callbacks, and reach `item_information_matrix()` zero times. Existing semantic-range tests continue to require genuine supported NumPy scalar controls to reach the established ATA path.

## References

MITRE. (2026). *CWE-1287: Improper validation of specified type of input (Version 4.20).* Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0.* https://owasp.org/www-project-application-security-verification-standard/

Souppaya, M., Scarfone, K., & Dodson, D. (2022). *Secure Software Development Framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218

Booth, H., Ogata, M., Kent, K., Souppaya, M., & Dodson, D. (2025). *Secure Software Development Framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST SP 800-218 Rev. 1, Initial Public Draft). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218r1.ipd
