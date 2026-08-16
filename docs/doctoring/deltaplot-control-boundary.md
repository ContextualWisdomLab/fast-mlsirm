# Delta-plot control trust boundary

Issue #942 hardens the Python-to-Rust control boundary of `delta_plot()`
without moving Angoff Delta-plot arithmetic out of Rust.

## Defect and threat model

Before this slice, the public adapter materialized `responses` and `group`,
then established `threshold`, `alpha`, `fixed_threshold`, `extreme`,
`const_range`, `nr_add`, `purify`, and `max_iter` with membership tests and
generic `int(...)` / `float(...)` conversions. Python string or numeric
subclasses and arbitrary conversion-protocol providers could therefore
execute caller code while semantic controls were being established. Invalid
controls also reached data materialization before package-owned rejection.

The correction uses an allow-known-good boundary. Exact built-in strings are
required for selector vocabulary. Exact built-in `int`/`float` values and the
package-supported concrete NumPy integer/floating scalar identities are
normalized once to built-ins. Booleans, Python/NumPy subclasses, and
arbitrary `__int__`, `__index__`, `__float__`, or comparison providers are
rejected without invoking caller callbacks. `alpha` must lie in `(0, 1)`,
`fixed_threshold` must be finite, `const_range` must be an exact two-tuple
satisfying `0 < lo < hi < 1`, `nr_add` must be a positive integer, and
`max_iter` lies in `1..MAX_MAX_ITER` so a caller cannot request an unbounded
purification budget through this adapter.

Accepted built-in controls then flow into the unchanged PyO3 seam. Response
and group validation, Angoff proportions, delta transforms, major-axis fit,
purification, thresholds, DIF flags, and every result-affecting calculation
remain in the existing Rust implementation.

## Verification design

The RED regression commit introduces executable hostile string/numeric
subclasses and protocol providers, response and group materialization
sentinels, and a native-core discovery sentinel. GREEN requires all rejected
controls to fail before every sentinel and with zero callback invocations.
Separate dispatch evidence verifies supported NumPy scalars and exact
selector strings reach the native seam as exact Python `str`/`float`/`int`
values. Hosted exact-head CI, security, coverage, package, provenance, and
review evidence remains mandatory before integration.

This evidence is intentionally about trust-boundary behavior, not statistical
validity. Existing Rust tests and Magis and Facon (2014) continue to own the
Delta-plot formula and purification behavior.

## Standards trace

The boundary follows MITRE CWE-1287's allow-known-good recommendation for
inputs expected to have a specified type. OWASP ASVS 5.0.0 remains the
current released ASVS baseline (released May 30, 2025). NIST SP 800-218 SSDF
1.1 remains the current final general SSDF; SP 800-218 Rev. 1 / SSDF 1.2 was
still an Initial Public Draft in the NIST publication registry when this
evidence was refreshed, so this repository does not present the draft as a
final requirement.

### References (APA 7th ed.)

CWE Content Team. (2026). *CWE-1287: Improper validation of specified type of
input* (CWE Version 4.20). MITRE.
https://cwe.mitre.org/data/definitions/1287.html

Magis, D., & Facon, B. (2014). deltaPlotR: An R package for differential item
functioning analysis with Angoff's delta plot. *Journal of Statistical
Software, 59*(Code Snippet 1), 1-19. https://doi.org/10.18637/jss.v059.c01

OWASP Foundation. (2025). *OWASP Application Security Verification Standard*
(Version 5.0.0).
https://owasp.org/www-project-application-security-verification-standard/

Scarfone, K., Souppaya, M., & Dodson, D. (2022). *Secure Software Development
Framework (SSDF) version 1.1: Recommendations for mitigating the risk of
software vulnerabilities* (NIST Special Publication 800-218). National
Institute of Standards and Technology.
https://doi.org/10.6028/NIST.SP.800-218
