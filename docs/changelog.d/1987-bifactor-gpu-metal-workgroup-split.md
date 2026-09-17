### Fixed

#### Bifactor GPU E-step Metal/WebGPU workgroup-dimension limit

- Split bifactor reduced E-step compute dispatches across `(x, y, z)` using the
  adapter's runtime `max_compute_workgroups_per_dimension` so Apple Metal no
  longer panics when a 1-D workgroup count exceeds 65535 (AC late-life
  multigroup bootstrap at q=241 required 141573 groups on `reduce_counts_blk`).
- Query `max_storage_buffer_binding_size` / `max_buffer_size` before allocating
  E-step buffers and fall back to the f64 CPU path when they do not fit; no
  hardcoded workgroup or byte caps.
- Extend study-precision CPU/GPU parity coverage with a q=481 leg gated by
  `STAGE5_HIGH_Q=1` alongside the existing 121/241 tests.
