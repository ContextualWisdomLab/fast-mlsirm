# GPU two-tier expected raw scores

## Added

- `expected_raw_two_tier_grm(..., device="cpu")` accepts `"gpu"` and `"auto"`.
  The GPU path computes weighted item/node f32 contributions with wgpu and
  uses host f64 accumulation. It is hybrid, not all-GPU reduction. The default
  stays on the CPU. `"gpu"` warns on CPU fallback; `"auto"` is silent. No usable
  adapter, insufficient device bounds or unsafe predictor precision can cause
  fallback, without reducing the caller's node count. Each contribution/readback
  buffer uses approximately `rows * items * nodes * 4` bytes per chunk, in
  addition to other buffers and host allocations. This materialized path can
  increase memory and transfer cost; no general speedup is claimed. The Python
  result does not expose Rust's `used_gpu` flag. A GPU request or equal CPU
  output is not a hardware-dispatch certificate. Finite inputs alone do not
  guarantee f32 accuracy.

## Changed

- The expected raw total is now computed in closed form,
  `Σ_i Σ_q w_q Σ_k σ(η_i + β_ik)`, which follows from linearity of
  expectation, instead of taking the mean of the Lord-Wingersky score
  distribution. The two agree within 1e-12 on the retained comparison fixtures;
  this is not a universal floating-point error bound. The recursion stays
  available as the verification oracle, and the per-row cost drops from
  quadratic to linear in the number of items.
