# Continuous-response-model control callback safety

## Scope

The public `fit_crm()` API marshals caller controls for a Rust-owned Samejima continuous response model. This boundary change does not alter the logit transform, Gauss-Hermite nodes or weights, observed-data likelihood, EM posterior bookkeeping, weighted-least-squares item M-step, convergence rule, identification/sign convention, derived discrimination/difficulty parameters, EAP scoring, estimands, or interpretation. Production mathematical and psychometric arithmetic remains in `mlsirm-core`.

Protected-main behavior discovered the compiled core before validating scalar controls. `max_iter` was compared directly before trusted identity was established, `tol` was sent through NumPy finiteness and comparison operations, and `q_theta`/`max_iter`/`tol` were normalized with `int(...)` or `float(...)` only at native dispatch. Caller-defined Python or NumPy scalar subclasses can override these conversion/comparison/ufunc hooks, so nominal numeric controls could execute caller code before the PyO3 boundary.

The corrected boundary validates controls before native-core discovery. Exact built-in Python integers and exact supported NumPy integer scalar classes are accepted for integer controls; exact built-in numerics and supported genuine NumPy numeric scalar classes are accepted for the tolerance. Subclasses and booleans are rejected before conversion. `q_theta` is constrained to the exact embedded Rust Gauss-Hermite domain `{7, 11, 15, 21, 31, 41}`, `max_iter` retains the existing repository bound, and `tol` retains the Rust finite strictly-positive contract. Only admitted controls are normalized to built-in values for PyO3 dispatch.

## Security and reliability interpretation

This is specified-input-type validation and trust-boundary hardening, not a new psychometric estimator. Exact runtime type identity prevents caller-controlled subclass hooks from becoming part of control validation. Validation errors do not interpolate rejected objects, so hostile representation callbacks are not required to produce package-owned evidence. Unsupported quadrature orders now fail before compiled-core discovery while preserving the same domain already enforced by Rust.

NIST SP 800-218 recommends secure design and validation of externally supplied inputs before potentially dangerous processing. That guidance is used as secure-development context only; this bounded change does not claim certification or conformance.

## Verification contract

Regression evidence must establish that:

- hostile Python integer, NumPy-integer, and floating subclasses fail before conversion, comparison, ufunc, representation, equality, or hash callbacks execute;
- malformed trusted controls fail before native-core discovery;
- exact Python booleans are not accepted as numeric CRM controls;
- genuine supported NumPy controls are normalized to built-in values before the PyO3 call;
- every Rust-embedded Gauss-Hermite order (`7`, `11`, `15`, `21`, `31`, `41`) remains available and unsupported orders fail closed;
- `max_iter` remains bounded by the existing repository limit;
- zero, negative, non-finite, and non-representable tolerances fail as package-owned validation errors;
- CRM mathematical results remain governed by existing Rust estimator, convergence, recovery, and backend evidence rather than new Python arithmetic.

## References

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological), 39*(1), 1–22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

Samejima, F. (1973). Homogeneous case of the continuous response model. *Psychometrika, 38*(2), 203–219. https://doi.org/10.1007/BF02291114

Wang, T., & Zeng, L. (1998). Item parameter estimation for a continuous response model using an EM algorithm. *Applied Psychological Measurement, 22*(4), 333–344. https://doi.org/10.1177/014662169802200402

Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. *The Annals of Statistics, 11*(1), 95–103. https://doi.org/10.1214/aos/1176346060
