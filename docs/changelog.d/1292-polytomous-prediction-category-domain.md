# Polytomous prediction category-domain admission

## Fixed

GRM/GPCM prediction admission now enforces the fitter-supported `2..=64` category domain at both the public Python boundary and direct Rust `polytomous_predictions()` boundary. Manually constructed `PolytomousFit` evidence above 64 categories fails before NumPy/native work, while direct native requests above `POLY_MAX_CAT` fail before prediction-grid allocation or item-parameter validation. Probability and expected-score arithmetic remain Rust-owned and unchanged.
