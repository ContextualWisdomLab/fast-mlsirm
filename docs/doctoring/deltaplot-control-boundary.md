# Delta-plot control trust boundary

Issue #942 hardens the Python-to-Rust control boundary of `delta_plot()` without moving Angoff Delta plot computation out of Rust.

## Defect and boundary

Before this slice, the public wrapper materialized caller-owned response and group arrays before validating semantic controls. Selector checks could invoke string-subclass comparison behavior, while `alpha`, `fixed_threshold`, `const_range`, `nr_add`, and `max_iter` used generic Python conversion/comparison paths before a package-trusted scalar identity was established. Rejected controls therefore could execute caller callbacks or consume data-marshalling work before fail-closed rejection.

The correction performs allow-known-good admission before either data materialization or compiled-core discovery. Selector controls require exact built-in strings. Numeric controls accept exact built-in numeric identities and the package-supported concrete NumPy integer/floating scalar identities; booleans, scalar subclasses, and arbitrary conversion providers are rejected before coercion. Constraint ranges additionally require an exact built-in two-element tuple so tuple-subclass indexing cannot run at the trust boundary. Every public control is type-admitted on that path, including fields that the active `threshold` / `extreme` branch will not consume, so a hosted options blob cannot smuggle a hostile unused field past the gate. Huge exact integers that overflow `float()` become package-owned `ValueError`s, matching the ICC control-extremes contract.

The normalized domains mirror the Rust core where applicable: normal-threshold `alpha` is finite in `(0, 1)`, constraint adjustment requires finite `0 <= lo < hi <= 1`, and additive adjustment requires `nr_add >= 1`. Fixed thresholds are finite because non-finite absolute thresholds cannot yield meaningful finite flag boundaries. `max_iter` is normalized to a built-in integer and bounded to `1..MAX_MAX_ITER` as a package resource-control ceiling.

After trusted controls are established, the existing response/group validation and PyO3 dispatch remain in place. Rust continues to own proportion calculation, extreme-proportion adjustment, Angoff delta transforms, covariance and major-axis calculation, threshold computation, iterative purification, convergence state, DIF flags, and all result-affecting arithmetic. Python performs validation and marshalling only.

## Verification design

The RED regression commit `be1cdf12f903b3d63379e4bcc84cdb9a009d66de` introduces hostile string, integer, float, and tuple subclasses, together with data-materialization and compiled-core sentinels. GREEN requires rejected controls to produce zero caller callbacks and to fail before either sentinel. Follow-up RED cases cover unused-branch hostiles, non-tuple unused `const_range` values, hostile elements inside an exact 2-tuple, `alpha=1.0`, wrong-length ranges, and huge exact integers that overflow `float()`. A fake native seam separately verifies genuine supported NumPy scalar controls, including the additive and fixed-threshold branches, are normalized to exact built-in values before PyO3 dispatch. Hosted exact-head CI, coverage, package, security, provenance, and formal review evidence remain required before lifecycle promotion.

This is trust-boundary and resource-control evidence, not new statistical-validity evidence. The existing Rust implementation and its pinned deltaPlotR/NumPy-oracle evidence continue to govern algorithmic parity and scientific behavior.

## Standards trace

The engineering boundary follows CWE-1287's allow-known-good guidance for specified input types. OWASP ASVS 5.0.0 is the current released ASVS baseline. NIST SP 800-218 SSDF 1.1 remains the current final general SSDF baseline; SP 800-218 Rev. 1 / SSDF 1.2 is tracked as draft rather than represented as final authority.

### References (APA 7th ed.)

CWE Content Team. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). MITRE. https://cwe.mitre.org/data/definitions/1287.html

OWASP Foundation. (2025). *OWASP Application Security Verification Standard* (Version 5.0.0). https://owasp.org/www-project-application-security-verification-standard/

Scarfone, K., Souppaya, M., & Dodson, D. (2022). *Secure Software Development Framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-218
