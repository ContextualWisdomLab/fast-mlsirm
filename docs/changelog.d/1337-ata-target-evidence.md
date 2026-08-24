# ATA target-curve evidence admission

## Fixed

- Reject callback-bearing, non-real, complex, and binary64-lossy target-theta or target-information evidence before NumPy materialization or item-information work, while preserving trusted NumPy/built-in numeric target curves and the historical single-point scalar target-information contract.
- Preserve exact real-numeric NumPy array rows nested inside inert built-in target trees without reopening array-provider callbacks; nested arrays are charged by logical size and replayed for lossless binary64 identity before materialization.
- Bound trusted ATA target evidence at 20,000,000 logical cells, built-in target nesting at 64 levels, and the dense target-point × item information matrix at 20,000,000 cells before per-cell conversion, NumPy materialization, psychometric scoring, or dense allocation; built-in tree traversal now keeps transient state proportional to nesting depth and bounds malformed zero-cell fan-out.
- Reuse the already-validated target-theta grid inside assembly so near-boundary evidence is not Python-level scanned a second time before item-information evaluation; the public item-information-matrix API still validates independent caller input.
