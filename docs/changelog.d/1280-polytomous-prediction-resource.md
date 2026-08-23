# Bound public polytomous prediction resources

## Fixed

- Reject public GRM/GPCM prediction grids above 20,000,000 dense probability cells before compiled-core discovery or output allocation.
- Keep GRM/GPCM category-probability and expected-score arithmetic in the existing Rust implementation; the Python guard is resource admission only.
- Track the remaining direct private-PyO3/core allocation bound in issue #1280 rather than treating the public guard as full native completion.
