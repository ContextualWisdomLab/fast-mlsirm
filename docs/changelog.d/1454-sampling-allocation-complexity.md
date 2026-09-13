# Bounded capped-strata allocation

## Fixed

- Replace repeated full active-set rescans in finite-population capped stratum allocation with a threshold-sorted water-filling pass, so the cap phase inspects each admitted stratum at most once after sorting.
- Add a maximum-envelope 100,000-strata census regression and an operation-count proof for the cap phase without relying on wall-clock timing.
- Preserve the existing Rust-owned proportional/Neyman quotas, census caps, deterministic input-order tie behavior, largest-remainder integerization, exact inclusion-probability ratios, and fail-closed zero-allocation contract.
