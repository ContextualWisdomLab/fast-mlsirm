# GPU two-tier expected raw scores

## Added

- `expected_raw_two_tier_grm(..., device="cpu")` accepts `"gpu"` and `"auto"`.
  The GPU path runs a wgpu f32 kernel with one thread per person or primary
  grid row. It uses compensated summation and dispatches rows in chunks, so
  device memory does not grow with the row count. `"gpu"` warns before falling
  back to the CPU when no adapter is usable. `"auto"` falls back silently. The
  default stays on the CPU.

## Changed

- The expected raw total is now computed in closed form,
  `Σ_i Σ_q w_q Σ_k σ(η_i + β_ik)`, which follows from linearity of
  expectation, instead of taking the mean of the Lord-Wingersky score
  distribution. The two agree to 1e-12. The recursion stays available as the
  verification oracle, and the per-row cost drops from quadratic to linear in
  the number of items.
