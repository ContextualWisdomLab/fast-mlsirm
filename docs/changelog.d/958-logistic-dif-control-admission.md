# Logistic DIF semantic-control admission

## Fixed

- Validate and normalize logistic-regression DIF and purified-DIF semantic controls before caller-owned response/group materialization and before compiled-core discovery.
- Reject callback-bearing scalar subclasses/protocol providers and booleans-as-numbers while preserving concrete supported NumPy scalar controls.
- Preserve Rust-owned DIF, purification, Benjamini-Hochberg, effect-size, convergence, and result arithmetic unchanged.
