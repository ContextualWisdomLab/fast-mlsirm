# Polytomous prediction category-domain admission

## Fixed

Public GRM/GPCM prediction admission now rejects manually constructed `PolytomousFit` evidence whose category-parameter shape implies more than the fitter-supported 64 categories before NumPy materialization or compiled-core discovery. Probability and expected-score arithmetic remain Rust-owned; direct native prediction-domain alignment remains tracked by issue #1292.
