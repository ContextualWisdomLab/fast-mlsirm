# Bradley-Terry control trust boundary

Issue #940 hardens the Python-to-Rust control boundary of `bradley_terry_mm()` without moving Bradley-Terry estimation arithmetic out of Rust.

## Defect and threat model

Before this slice, the public Python wrapper materialized `wins`, discovered the compiled core, and only then called `float(alpha)`, `int(max_iter)`, and `float(tol)`. Python numeric subclasses and arbitrary conversion-protocol providers could therefore execute caller code while semantic controls were being established. Exact-domain failures such as non-finite `alpha`/`tol` or a zero iteration budget were left to the lower native layer after those side effects.

The correction uses an allow-known-good boundary. Exact built-in `int`/`float` values and the package-supported concrete NumPy integer/floating scalar identities are normalized once to built-ins. Booleans, Python/NumPy subclasses, and arbitrary `__int__`, `__index__`, or `__float__` providers are rejected without invoking caller callbacks. `alpha` must be finite and non-negative, `tol` finite and positive, and `max_iter` lies in `1..MAX_MAX_ITER` so a caller cannot request an unbounded MM iteration budget through this wrapper. Exact integers that overflow IEEE-754 conversion (`10**10000`) raise the same package-owned `ValueError` used by the ICC adapter (`"{name} must be finite"`), not a raw `OverflowError`.

The installed wrapper runs before the historical `scaling.bradley_terry_mm` implementation. Accepted built-in controls then flow through the unchanged wrapper into PyO3; win-matrix validation and all MM likelihood, worth update, normalization, convergence, and result calculations remain in the existing Rust implementation.

## Verification design

The RED regression commit introduces executable hostile float/integer subclasses and protocol providers, a win-matrix materialization sentinel, and a native-core discovery sentinel. GREEN requires all rejected controls to fail before every sentinel and with zero callback invocations. Separate dispatch evidence verifies supported NumPy scalars reach the native seam as exact Python `float`/`int` values. Hosted exact-head CI, security, coverage, package, provenance, and review evidence remains mandatory before integration.

This evidence is intentionally about trust-boundary behavior, not statistical validity. The Bradley–Terry paired-comparison model is traced to Bradley and Terry (1952), and the MM algorithm family used for generalized Bradley–Terry estimation is traced to Hunter (2004). These method sources document the unchanged statistical computation; this slice does not claim new estimation or validity properties.

## Standards trace

The boundary follows MITRE CWE-1287's allow-known-good recommendation for inputs expected to have a specified type. OWASP ASVS 5.0.0 remains the current released ASVS baseline (released May 30, 2025). NIST SP 800-218 SSDF 1.1 remains the current final general SSDF; SP 800-218 Rev. 1 / SSDF 1.2 was still an Initial Public Draft in the NIST publication registry when this evidence was refreshed, so this repository does not present the draft as a final requirement.

### References (APA 7th ed.)

Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345. https://doi.org/10.1093/biomet/39.3-4.324

CWE Content Team. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). MITRE. https://cwe.mitre.org/data/definitions/1287.html

Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models. *The Annals of Statistics, 32*(1), 384–406. https://doi.org/10.1214/aos/1079120141

OWASP Foundation. (2025). *OWASP Application Security Verification Standard* (Version 5.0.0). https://owasp.org/www-project-application-security-verification-standard/

Scarfone, K., Souppaya, M., & Dodson, D. (2022). *Secure Software Development Framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218
