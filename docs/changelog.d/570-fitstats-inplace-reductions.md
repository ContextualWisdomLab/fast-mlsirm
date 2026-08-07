# Bounded NumPy fallback infit and outfit reductions

## Changed

- Reused the existing masked squared-residual buffer for the NumPy fallback
  outfit division and used a Boolean `where=` reduction for the infit variance
  denominator, avoiding a full float mask copy and a separate full division
  result.
- Preserved the compiled Rust primary path, statistical equations, probability
  clipping, missing-response handling, public APIs, and model identities.
- Added sparse-missingness, all-missing-item, extreme-probability, source-level
  allocation, reproducible benchmark, and APA 7 doctoring evidence without a
  universal performance claim.
