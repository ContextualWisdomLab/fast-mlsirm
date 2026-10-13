# Two-tier GRM GPU E-step

## Added

- Two-tier GRM fits accept `device="gpu"` or `"auto"` to run the EM
  E-step on the GPU; the M-step stays on the CPU. Results agree with the
  CPU fit within single-precision tolerance, including fits with missing
  responses and specific-free items. On an Apple Metal adapter, a
  two-primary fit (300 persons, 11-point grids) ran 5.9-7.1 times faster.
  `"cpu"` remains the default. `"gpu"` warns and falls back to the CPU when
  no adapter is available or the problem exceeds the adapter's buffer
  limits (#2282).

## Fixed

- The shared GPU reduced E-step kernel now writes each person-block
  observation flag (`anyobs`) from a single invocation instead of one per
  general node. The old stores wrote the same value, so results do not
  change; the change removes a non-atomic multi-writer store (#2282).
