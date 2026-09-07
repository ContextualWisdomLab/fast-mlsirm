# Marginal EAP binary64 reduction contract

## Changed

- Added a deterministic unit regression proving that reassociating marginal EAP posterior-weight accumulation from the established multiply-then-reduce route into a ternary `np.einsum` changes an ordinary finite `float64` moment by one ULP.
- Restored `python/fast_mlsirm/estimators/marginal.py` and the Bolt guidance to protected-main semantics; this change does not authorize a new Python numerical hot path. Future material optimization requires controlled profiling, deterministic CPU-f64 parity, realistic estimator/recovery evidence, and should prefer the canonical Rust numerical owner.
