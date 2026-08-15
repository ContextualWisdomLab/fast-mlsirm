# Parallel-analysis control callback safety

## Scope

The public parallel-analysis contract accepts three integer controls: the random-simulation count, the comparison centile, and the deterministic random seed. This boundary is validation and marshalling only. Observed and random eigenvalue calculation, bias adjustment, the leading-component retention scan, the deterministic random stream, and every other result-affecting operation remain Rust-owned and unchanged.

Protected `main` previously accepted broad Python and NumPy integer subclasses and then invoked `int(...)`. A caller-defined subclass can override integer conversion, so nominal integer acceptance could execute caller-controlled code before the package established a trusted control value. The wrapper also discovered the compiled native core before validating explicit controls, allowing malformed control metadata to cross the native-loader trust boundary before rejection.

The corrected boundary uses positive validation. Exact built-in Python integers are accepted directly; exact supported NumPy integer scalar classes are normalized after exact-type admission; booleans, integer subclasses, and arbitrary conversion providers are rejected without conversion or representation callbacks. Explicit `n_iterations`, `centile`, and `seed` values are validated before native-core discovery.

## Preserved scientific contract

The correction does not change the implemented Horn/Glorfeld parallel-analysis method or claim a universal factor-count rule. The existing contract remains:

- omitted `n_iterations` resolves to `30 * n_items`;
- explicit `n_iterations` is a positive integer;
- `centile=0` uses the mean random eigenvalue benchmark and `1..=99` selects the corresponding upper centile;
- `seed` must fit the Rust/PyO3 unsigned 64-bit transport;
- the random-eigenvalue workspace is bounded to 128 MiB before Rust dispatch; and
- factor retention remains distinct from structural measurement-model selection under #608.

## Security and reliability interpretation

MITRE CWE-1287 recommends validating that input has the expected type and using an accept-known-good strategy. This correction therefore establishes a closed set of trusted scalar identities instead of attempting to detect hostile conversion behavior after execution.

NIST SP 800-218, Secure Software Development Framework Version 1.1, remains the final secure-development baseline used by this repository. NIST SP 800-218 Rev. 1 / SSDF Version 1.2 is an Initial Public Draft as of this record and is tracked as draft guidance rather than represented as final. OWASP ASVS 5.0.0 input-validation guidance is informative for this caller-controlled boundary; this package does not claim ASVS certification or conformance from this bounded change.

## Verification contract

Regression evidence must prove all of the following:

- exact built-in Python integers remain accepted;
- exact supported NumPy integer scalars remain accepted and are normalized to built-in integers;
- Python and NumPy integer subclasses are rejected before hostile `__int__` or `__repr__` callbacks execute;
- malformed explicit controls fail before native-core discovery;
- positive-count, centile, seed-range, and random-workspace errors remain package-owned and bounded; and
- the accepted-control path still delegates to the same Rust `parallel_analysis` entrypoint with unchanged values.

## References

Dinno, A. (2018). *paran: Horn's test of principal components/factors* (Version 1.5.6) [R package]. Comprehensive R Archive Network. https://CRAN.R-project.org/package=paran

Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement, 55*(3), 377–393. https://doi.org/10.1177/0013164495055003002

Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179–185. https://doi.org/10.1007/BF02289447

MITRE. (2026). *CWE-1287: Improper validation of specified type of input* (Version 4.20). Common Weakness Enumeration. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software development framework (SSDF) version 1.2: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218 Rev. 1, Initial Public Draft). https://doi.org/10.6028/NIST.SP.800-218r1.ipd

OWASP Foundation. (2025). *OWASP Application Security Verification Standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/
