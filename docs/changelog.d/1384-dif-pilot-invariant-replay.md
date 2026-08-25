# DIF pilot invariant replay

## Fixed

- Replay `DifPilotDesign` reference/focal group, row-alignment, identifier, and schema invariants before group-array construction, observed-score DIF argument projection, and canonical serialization/fingerprinting so frozen-record rebinding cannot silently change the populations supplied to DIF analysis.
- Require the wrapped binary pilot design to be the exact package-owned `MirtPilotDesign` record and replay group assignments only from an exact inert tuple, preventing caller-defined record/container subclasses from executing field, length, or iteration callbacks at the handoff boundary.

Production Mantel-Haenszel, logistic DIF, SIBTEST, purification, effect-size, and significance arithmetic remains unchanged and Rust-owned.
