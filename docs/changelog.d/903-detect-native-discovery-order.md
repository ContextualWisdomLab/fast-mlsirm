# Validate DETECT inputs before native discovery

## Fixed

- Validate and marshal DETECT and DIMTEST public response/partition inputs before compiled-core discovery, so rejected requests remain package-owned validation failures without crossing the native-loader boundary while all result-affecting dimensionality arithmetic remains Rust-owned.
