# ATA target-curve evidence admission

## Fixed

- Reject callback-bearing, non-real, complex, and binary64-lossy target-theta or target-information evidence before NumPy materialization or item-information work, while preserving trusted NumPy/built-in numeric target curves and the historical single-point scalar target-information contract.
- Bound trusted ATA target evidence at 20,000,000 logical cells and the dense target-point × item information matrix at 20,000,000 cells before per-cell conversion, NumPy materialization, psychometric scoring, or dense allocation; built-in tree traversal now keeps transient state proportional to nesting depth and bounds malformed zero-cell fan-out.
