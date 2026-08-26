# Finite-population proportion sampling design

## Added

- Added a domain-neutral Rust/PyO3 `fast-mlsirm.sampling-design.v1` contract for normal-approximation finite-population proportion sample size, finite-population correction, and caller-selected proportional or equal-cost Neyman stratum allocation. Python only validates and marshals exact caller evidence; sample-size, correction, and allocation arithmetic remains Rust-owned.

## Fixed

- Replay the Rust `population_size <= 2^53` and 100,000-strata resource domains at the Python boundary before member normalization or Rust dispatch, and reject integer-valued strict probability controls without an unnecessary `float(...)` conversion so oversized/invalid controls fail with package-owned `ValueError` rather than consuming avoidable work or surfacing conversion overflow.
