# Plausible-value serving control safety

## Fixed

- Validate and normalize public plausible-value `n_draws`, `seed`, and `device` controls before compiled-core discovery.
- Bound `seed` to the Rust/PyO3 `u64` contract, keep `n_draws` within the existing serving limit, and constrain device selection to `cpu`, `gpu`, or `auto`.
- Reject booleans, caller-defined integer/string subclasses, arbitrary coercion providers, and hostile scalar metaclasses without executing their conversion, hashing, or equality callbacks.
- Preserve exact supported NumPy integer scalar compatibility by admitting trusted scalar types through identity-only comparisons and marshalling them once to built-in integers.

Closes #914.
