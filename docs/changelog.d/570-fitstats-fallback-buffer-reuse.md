# Bounded fit-statistics fallback buffers

## Changed

- Reused one owned NumPy squared-residual buffer for fallback infit/outfit calculations, retained the infit numerator before in-place division, and used a Boolean `where=` reduction for the variance denominator without a full numeric mask copy.
- Added deterministic sparse-missingness and clipped-probability parity contracts plus an environment-specific benchmark while retaining Rust as the production fit-statistics backend.
