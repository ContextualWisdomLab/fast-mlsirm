# WLE complex-evidence admission

## Fixed

- Reject complex-valued dichotomous and polytomous WLE responses and item parameters before real-valued marshalling or Rust scoring dispatch, preventing imaginary components from being silently discarded.