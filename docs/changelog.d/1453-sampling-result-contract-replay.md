# Sampling result contract replay

## Fixed

- Replay the Rust-owned sampling algorithm identity before public result marshalling so a same-schema stale or foreign extension fails closed instead of being exposed under the current contract.
- Validate every returned stratum inclusion-probability ratio against the Rust-returned sample and population counts before constructing the public sampling artifact, with stable package-owned errors for missing, malformed, count-mismatched, or inconsistent ratio evidence.
- Preserve the Rust-owned sample-size, finite-population-correction, proportional-allocation, and equal-cost Neyman arithmetic unchanged.
