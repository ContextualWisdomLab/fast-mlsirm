# Configuration integer callback safety

## Problem

Public configuration validation accepted Python's generic integer protocol. Calling `operator.index()` or comparing caller-controlled integer-like objects before trust was established allowed arbitrary `__index__` implementations or integer subclasses to participate in validation. Construction-time validation closed the memory-budget bypass, but `seed` and `verbose` still stored untrusted objects, and admitted narrow NumPy integers were left on the frozen fields. Those stored scalars wrap on ordinary Python/NumPy arithmetic: `np.uint8(16) * np.uint8(16)` becomes `0`, so `n_items` and `simulate()` can disagree with the validated size product, and `config.seed + restart` can wrap a restart index.

## Boundary decision

Configuration validation is marshalling and trust-boundary work, not psychometric arithmetic. The package now accepts only exact built-in `int` values and exact supported NumPy integer scalar types for validated integer controls. Accepted NumPy scalars are converted to built-in integers for bounds and work-budget calculations; booleans, caller-defined `int` subclasses, and arbitrary index providers are rejected without invoking their coercion hooks. After those checks pass, the trusted built-in integers are written back onto the frozen dataclass so later size products, RNG seeding, and save-time `int(seed)` cannot dispatch caller callbacks or wrap.

`MLS2PLMConfig` and `FitConfig` run that same validator from `__post_init__`, so invalid or untrusted integer controls cannot exist as constructed objects. `validate()` remains public and idempotent for callers that already invoke it at simulate/fit entry points.

The hardened surface covers simulation sizes, latent dimension, and simulation `seed`, plus fit latent dimension, optimizer iteration/restart/history controls, quadrature node counts, marginal M-step count, latent-space integration point/seed controls, fit `seed`, and `verbose`. The same trusted-integer marshalling is applied to `dimensionality_diagnostics` `k_folds`, `seed`, and each `latent_dims` candidate before the diagnostic fit-budget product or `seed + fold_idx` can wrap a narrow NumPy scalar, and to `fit_diagnostics` `parameter_count` and `m2_q_*` before AIC/BIC arithmetic or save-time `int(q_*)` can dispatch caller `__index__` hooks. Numerical model ownership and Rust-first computation are unchanged.

## Test evidence

`tests/test_config_integer_callback_safety.py` provides hostile `__index__` regressions, valid-valued caller `int` subclasses, genuine NumPy scalar compatibility, write-back of narrow unsigned sizes, a 2-by-256 simulation shape check that would fail if `n_items` wrapped, dimensionality-diagnostic wrap/`__index__` checks for `k_folds`, fold `seed`, and `latent_dims`, and `fit_diagnostics` `__index__` checks for `parameter_count` and `m2_q_*`. The original RED commit is `4c81e4dc465312d13f044b9b47e14d839af6cc1a`; exact-head hosted CI/security/package/coverage/review evidence remains authoritative as the branch advances.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

MITRE. (n.d.). *CWE-1287: Improper validation of specified type of input*. https://cwe.mitre.org/data/definitions/1287.html

National Institute of Standards and Technology. (2022). *Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

OWASP Foundation. (2025). *OWASP application security verification standard 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

Python Software Foundation. (n.d.). Emulating numeric types. In *The Python language reference*. https://docs.python.org/3/reference/datamodel.html#object.__index__
