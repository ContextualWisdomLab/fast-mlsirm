# Polytomous fit semantic control safety

## Fixed

- Reject caller-defined text, integer, real, and hashing protocols before GRM/GPCM calibration controls are normalized, response data are materialized, or the Rust core is discovered.
- Require calibration quadrature to use an exact supported integer node count rather than callback-capable membership or lossy coercion.
- Normalize both `NaN` and `-1` as missing polytomous responses before category validation, and report malformed response conversion through a stable package-owned numeric-input error.
- Preserve the category, iteration, and positive-finite stopping contracts while keeping the Bock-Aitkin EM/Newton estimator and all result-affecting psychometric arithmetic Rust-owned.
