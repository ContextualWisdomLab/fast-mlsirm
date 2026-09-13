# Marginal EAP binary64 reduction contract

## Fixed

- Added a deterministic scalar IEEE-754 binary64 regression proving that the contraction order selected by reassociating marginal EAP posterior-weight accumulation can change an ordinary finite moment by one ULP. The oracle forces an explicit binary64 rounding boundary after each operation instead of treating backend-dependent optimized NumPy/FMA output as the reference.
- Restored `python/fast_mlsirm/estimators/marginal.py` and the Bolt guidance to protected-main semantics; this change does not authorize a new Python numerical hot path. Future material optimization requires controlled profiling, deterministic CPU-f64 parity, realistic estimator/recovery evidence, and should prefer the canonical Rust numerical owner.
