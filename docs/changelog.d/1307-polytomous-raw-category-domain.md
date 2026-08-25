# Polytomous raw prediction category-domain replay

## Fixed

- Replayed the fitter-supported `2..=64` category domain in the package-private Python prediction helper before resource calculation or compiled-core discovery, while preserving the valid 64-category boundary and keeping GRM/GPCM probability arithmetic Rust-owned.
