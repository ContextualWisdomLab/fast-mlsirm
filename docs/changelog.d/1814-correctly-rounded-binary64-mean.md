# Add a correctly-rounded finite binary64 mean primitive

## Changed

- Add the Rust-owned `fast_mlsirm.binary64_mean@1.0.0` numerical contract for non-empty finite binary64 slices. Inputs are accumulated exactly in `2^-1074` units and divided by the original count before one final round-to-nearest, ties-to-even projection.
- Preserve exact-zero identity separately from rounded-zero output, including signed nonzero underflow, cancellation near `f64::MAX`, subnormal/normal boundaries, and permutation-independent results.
- Add public regression evidence for the TEPP half-ULP mixed-sign counterexample, `[f64::MAX, 1e-16, -f64::MAX]`, and a deterministic 10,000-case subnormal exact integer/rational oracle without importing TEPP domain semantics or creating a mutable cross-repository dependency.
- Record the fixed 34×`u64` exact-accumulator decision, capacity proof, rejected floating-reduction alternatives, failure behavior, and immutable-consumer boundary in Proposed `ADR-0029`; it is not Accepted until protected integration and release evidence complete.
- TEPP consumption remains gated on a later immutable fast-mlsirm release.
