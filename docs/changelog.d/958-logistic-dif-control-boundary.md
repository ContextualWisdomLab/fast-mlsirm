# Harden observed-score logistic DIF controls

## Changed

- Validate logistic and purified observed-score DIF semantic controls before caller-owned response/group materialization and before compiled Rust-core discovery.
- Reject caller-defined scalar subclasses, arbitrary conversion providers, booleans-as-numbers, invalid FDR levels, zero iteration caps, negative anchor floors, and values outside native `usize` without invoking caller callbacks.
- Preserve genuine supported NumPy scalar compatibility and keep all logistic/Mantel-Haenszel/purification statistics and BH arithmetic Rust-owned.
