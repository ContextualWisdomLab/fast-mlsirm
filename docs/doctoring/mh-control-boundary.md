# Mantel-Haenszel DIF control trust boundary

Issue #949 hardens the Python-to-Rust control boundary of `mantel_haenszel_dif()` without moving Mantel-Haenszel computation out of Rust.

## Defect and boundary

Before this slice, the public wrapper discovered the compiled Rust core and materialized caller-owned response and group arrays before validating `fdr_q` and `exclude_studied_item`. `fdr_q` used generic finiteness and `float(...)` conversion after data work, and `exclude_studied_item` was forced through `bool(...)` at dispatch. Rejected controls therefore could execute caller callbacks or consume native discovery and data-marshalling work before fail-closed rejection.

The correction performs allow-known-good admission before either data materialization or compiled-core discovery. `exclude_studied_item` requires an exact built-in `bool`. `fdr_q` accepts exact built-in numeric identities and the package-supported concrete NumPy integer/floating scalar identities; booleans, scalar subclasses, and arbitrary conversion providers are rejected before coercion. Huge exact integers that overflow `float()` become package-owned `ValueError`s, matching the ICC and delta-plot control-extremes contract. The existing `fdr_q` domain remains finite `(0, 1]`. The ETS default still includes the studied item in the matching total (`exclude_studied_item=False`).

After trusted controls are established, the existing response/group validation and PyO3 dispatch remain in place. Rust continues to own common odds ratios, continuity-corrected chi-square, ETS delta, Robins-Breslow-Greenland standard errors, standardized P-DIF, ETS A/B/C classification, and Benjamini-Hochberg flagging. Python performs validation and marshalling only.

Purified MH, logistic DIF, and SIBTEST keep their current wrappers and are out of scope for this slice.

## Verification design

`tests/test_mh_control_callback_safety.py` introduces hostile float and bool-like subclasses together with data-materialization and compiled-core sentinels. GREEN requires rejected controls to produce zero caller callbacks and to fail before either sentinel. Additional RED cases cover domain-invalid `fdr_q`, boolean-as-number `fdr_q`, integer and NumPy-bool matching flags, and huge exact integers that overflow `float()`. A fake native seam verifies genuine supported NumPy `fdr_q` scalars and an exact `exclude_studied_item=True` flag normalize to built-in values before PyO3 dispatch. Hosted exact-head CI, coverage, package, security, provenance, and formal review evidence remain required before lifecycle promotion.

This is trust-boundary and resource-control evidence, not new statistical-validity evidence. The existing Rust implementation and its Holland and Thayer (1988) / Donoghue, Holland, and Thayer (1993) contract continue to govern algorithmic behavior.

## Standards trace

The engineering boundary follows CWE-1287's allow-known-good guidance for specified input types. OWASP ASVS 5.0.0 is the current released ASVS baseline. NIST SP 800-218 SSDF 1.1 remains the current final general SSDF baseline; SP 800-218 Rev. 1 / SSDF 1.2 is tracked as draft rather than represented as final authority.

### References (APA 7th ed.)

CWE Content Team. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). MITRE. https://cwe.mitre.org/data/definitions/1287.html

Donoghue, J. R., Holland, P. W., & Thayer, D. T. (1993). A Monte Carlo study of factors that affect the Mantel-Haenszel and standardization measures of differential item functioning. In P. W. Holland & H. Wainer (Eds.), *Differential item functioning* (pp. 137-166). Erlbaum.

Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel procedure. In H. Wainer & H. I. Braun (Eds.), *Test validity* (pp. 129-145). Erlbaum.

OWASP Foundation. (2025). *OWASP Application Security Verification Standard* (Version 5.0.0). https://owasp.org/www-project-application-security-verification-standard/

Scarfone, K., Souppaya, M., & Dodson, D. (2022). *Secure Software Development Framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218
