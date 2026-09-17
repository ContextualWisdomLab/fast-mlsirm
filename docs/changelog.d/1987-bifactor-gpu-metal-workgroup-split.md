### Fixed

#### Bifactor GPU E-step Metal/WebGPU workgroup-dimension limit

- Split bifactor reduced E-step compute dispatches across `(x, y, z)` using the
  adapter's runtime `max_compute_workgroups_per_dimension` so Apple Metal no
  longer panics when a 1-D workgroup count exceeds 65535 (AC late-life
  multigroup bootstrap at q=241 required 141573 groups on `reduce_counts_blk`).
- Query `max_storage_buffer_binding_size` / `max_buffer_size` before allocating
  E-step buffers and fall back to the f64 CPU path when they do not fit; no
  hardcoded workgroup or byte caps.
- Replace the WGSL zero-mass log-weight sentinel `-1e300` with an f32-representable
  `-1e37` so `create_shader_module` succeeds on Metal (WGSL rejects the abstract
  literal inside an `f32` comparison).
- Re-enable the `mlsirm-core` default `gpu` feature on the PyO3 cdylib (it had been
  disabled via `default-features = false` in a WIP salvage commit), so `device="gpu"`
  again reaches the wgpu kernels instead of always falling back to CPU.
- Extend study-precision CPU/GPU parity coverage with a q=481 leg and a wide-item
  q=241 Metal 2-D dispatch leg gated by `STAGE5_HIGH_Q=1`.
