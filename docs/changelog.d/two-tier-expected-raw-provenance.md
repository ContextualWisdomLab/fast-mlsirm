# Two-tier expected raw: device provenance

## Added

- `score_two_tier_grm_expected_raw(fit, specific_map, q_specific, *, device="cpu")`
  returns `TwoTierGrmExpectedRawScores(expected_raw, device_requested, used_gpu)`.
  It computes the same values as `expected_raw_two_tier_grm`, with the same
  validation and native arguments. It also reports the Rust dispatcher's
  `used_gpu` flag, so a `"gpu"` or `"auto"` request can be told apart from a
  silent CPU fallback. `used_gpu=True` means only that the weighted item/node
  f32 contributions ran on wgpu, followed by host f64 accumulation. It is not
  all-GPU reduction and not an accuracy certificate; the f32 bound from #2288
  still applies. Malformed native provenance, a CPU request that reports GPU
  execution, or a compiled core without the new binding raises `RuntimeError`
  instead of being coerced. The scoring estimand, quadrature counts and
  formulas are unchanged (Cai, 2015, pp. 542-543, Eqs. 14-17).
- The PyO3 binding `two_tier_expected_raw_with_provenance` returns
  `{"values", "used_gpu"}`. It shares one validation and dispatch helper with
  `two_tier_expected_raw`. `expected_raw_two_tier_grm` and the legacy binding
  keep their ndarray return, signature and native call.
