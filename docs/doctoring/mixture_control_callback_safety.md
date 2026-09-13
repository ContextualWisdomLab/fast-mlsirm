# Mixture IRT control callback safety

## Scope

The public `fit_mixture()` API marshals caller controls for a Rust-owned mixed Rasch / mixture-2PL estimator. This boundary change does not alter the mixture likelihood, quadrature, E-step responsibilities, Newton M-step, restart selection, convergence rule, class canonicalization, estimands, or interpretation. Production mathematical and psychometric arithmetic remains in `mlsirm-core`.

Protected-main behavior discovered the compiled core before validating scalar controls and accepted broad Python/NumPy integer subclasses before later `int(...)`, `float(...)`, or `str(...)` coercion. A caller-defined subclass can override those conversion or representation hooks, so nominal scalar acceptance could execute caller-controlled code before trusted control values were established.

The corrected boundary validates controls before native-core discovery. Exact built-in Python integers and exact supported NumPy integer scalar classes are accepted for bounded integer controls; exact built-in numeric values and supported NumPy numeric scalar classes are accepted for the finite non-negative tolerance; subclasses are rejected before conversion. The model selector accepts only an exact built-in string while preserving every alias already accepted by the Rust binding. The seed is bounded to the Rust/PyO3 `u64` domain before dispatch.

## Security and reliability interpretation

This is specified-input-type validation and trust-boundary hardening, not a new psychometric model. The allow-list is defined by exact runtime type identity rather than broad subclass membership, and trusted NumPy scalars are normalized only after admission. Rejected values are not interpolated into errors, so hostile `__repr__` hooks are not needed to explain validation failures. Resource-work and output-buffer limits remain unchanged.

NIST SP 800-218 recommends validating inputs and applying secure design practices before potentially dangerous processing. That guidance is used here as secure-development context only; this repository does not claim certification or conformance from this bounded change.

## Verification contract

Regression evidence must establish that:

- hostile Python integer, NumPy-integer, floating, and string subclasses fail before their conversion, normalization, representation, equality, or hash callbacks execute;
- invalid exact controls fail before native-core discovery;
- genuine supported NumPy controls are normalized to built-in values before the PyO3 call;
- the Rust binding's existing aliases (`rasch`, `Rasch`, `RASCH`, `2pl`, `2PL`, `twopl`, `TwoPl`) remain supported;
- zero tolerance remains valid, preserving the Rust reduction-anchor contract;
- negative/non-finite tolerance and out-of-range `u64` seeds fail closed;
- mixture mathematical results remain governed by the existing Rust tests and recovery evidence rather than by new Python arithmetic.

## References

Frick, H., Strobl, C., Leisch, F., & Zeileis, A. (2012). Flexible Rasch mixture models with package psychomix. *Journal of Statistical Software, 48*(7), 1–25. https://doi.org/10.18637/jss.v048.i07

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Rost, J. (1990). Rasch models in latent classes: An integration of two approaches to item analysis. *Applied Psychological Measurement, 14*(3), 271–282. https://doi.org/10.1177/014662169001400305

Rost, J., & von Davier, M. (1995). Mixture distribution Rasch models. In G. H. Fischer & I. W. Molenaar (Eds.), *Rasch models: Foundations, recent developments, and applications* (pp. 257–268). Springer. https://doi.org/10.1007/978-1-4612-4230-7_14
