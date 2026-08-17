# Judge category-count control safety

## Scope

Issue #912 hardens only the Python validation/marshalling boundary for the public `validate_judge(..., k=...)` category-count control. The Rust-owned agreement, quadratic-weighted kappa, degradation, standardized-mean-difference/fairness, gate-decision arithmetic, governed thresholds, and result schema are unchanged.

Protected `main` previously imported the compiled core before validating `k` and repeatedly invoked `int(k)`. Because Python integer subclasses and arbitrary integer-protocol objects may supply executable conversion methods, that order allowed caller-controlled conversion code to run while establishing a security- and fairness-relevant control.

The bounded correction admits only an exact built-in `int` or an exact concrete NumPy integer scalar identity, normalizes a trusted NumPy scalar once, enforces the existing `2..=1000` domain, completes label/policy marshalling, and only then imports the Rust core. Booleans, Python/NumPy subclasses, and arbitrary conversion-protocol providers fail before conversion callbacks or Rust dispatch. NumPy scalar-type admission uses identity comparisons rather than set membership so a caller-controlled scalar metaclass cannot inject `__hash__` or `__eq__` execution into the trust decision.

## Verification contract

The regression suite must prove all of the following on the exact PR head:

- hostile Python integer subclasses execute zero `__int__` callbacks;
- hostile NumPy integer subclasses execute zero `__int__` callbacks;
- arbitrary `__int__` and `__index__`-only providers execute zero callbacks;
- caller-controlled NumPy scalar metaclasses execute zero hashing/equality callbacks during type admission;
- booleans, `np.bool_`, and 0-d `ndarray` values are rejected as non-exact integer scalars;
- invalid exact category counts and type-invalid controls fail before compiled-core import;
- genuine built-in integers and concrete NumPy integer scalars remain compatible and arrive at the Rust boundary as an exact built-in integer;
- the existing `2..=1000` domain still rejects an exact in-type overflow; and
- no judge-validation numerical formula or policy threshold changes.

The initial test commit `7cf9eb6c2937020b5e755b5ae0a6cc2380fc068d` records the conversion/native-discovery RED contract, and `db0cb5848d317d39f197933043e289e00cdf522b` supplies its first bounded GREEN. A second RED at `3979f064b3e32fe44892cab369f8ffc1e3af4d73` demonstrates that hashed type-container membership would still execute caller-controlled metaclass hooks; `54fe33b2dd9d2a287c04635f2acba7bfc94f10fa` replaces that admission with identity-only comparisons. Hosted exact-head evidence remains authoritative over these remembered identities and must be refetched before lifecycle or integration decisions.

## Standards and research basis

The security references below govern input-validation and development-process evidence; they are not psychometric validity authorities. OWASP ASVS 5.0.0 is the latest stable ASVS release as checked on 2026-08-16; OWASP separately labels its bleeding-edge build as preview-only. NIST SP 800-218 Rev. 1 / SSDF 1.2 remains an Initial Public Draft, so this work retains final SSDF 1.1 as the normative NIST baseline and records the draft only as a standards-watch item.

### References (APA 7th)

MITRE. (2026). *CWE-1287: Improper validation of specified type of input* (CWE Version 4.20). Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). U.S. Department of Commerce. https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). U.S. Department of Commerce. https://csrc.nist.gov/pubs/sp/800/218/r1/ipd

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://github.com/OWASP/ASVS/tree/v5.0.0_release

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
