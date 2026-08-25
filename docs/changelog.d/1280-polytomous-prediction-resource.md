# Bound polytomous prediction resources

## Fixed

- Reject public GRM/GPCM prediction grids above 20,000,000 dense probability cells before compiled-core discovery or output allocation.
- Apply the same 20,000,000-cell ceiling inside the Rust `polytomous_predictions` owner before item-parameter validation or `Vec::with_capacity`, so direct core/PyO3 callers cannot bypass the public resource envelope.
- Keep GRM/GPCM category-probability and expected-score arithmetic in the existing Rust implementation; the added checks govern request size and allocation only.
