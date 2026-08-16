# Configuration integer callback safety

## Problem

Public configuration validation accepted Python's generic integer protocol. Calling `operator.index()` or comparing caller-controlled integer-like objects before trust was established allowed arbitrary `__index__` implementations or integer subclasses to participate in validation.

## Boundary decision

Configuration validation is marshalling and trust-boundary work, not psychometric arithmetic. The package now accepts only exact built-in `int` values and exact supported NumPy integer scalar types for validated integer controls. Accepted NumPy scalars are converted to built-in integers for bounds and work-budget calculations; booleans, caller-defined `int` subclasses, and arbitrary index providers are rejected without invoking their coercion hooks.

`MLS2PLMConfig` and `FitConfig` run that same validator from `__post_init__`, so invalid or untrusted controls cannot exist as constructed objects. `validate()` remains public and idempotent for callers that already invoke it at simulate/fit entry points.

The hardened surface covers simulation sizes and latent dimension plus fit latent dimension, optimizer iteration/restart/history controls, quadrature node counts, marginal M-step count, and latent-space integration point/seed controls. Numerical model ownership and Rust-first computation are unchanged.

## Test evidence

`tests/test_config_integer_callback_safety.py` provides hostile `__index__` regressions, valid-valued caller `int` subclasses, and genuine NumPy scalar compatibility. The original RED commit is `4c81e4dc465312d13f044b9b47e14d839af6cc1a`; exact-head hosted CI/security/package/coverage/review evidence remains authoritative as the branch advances.
