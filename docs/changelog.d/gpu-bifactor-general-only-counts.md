# GPU bifactor E-step keeps general-only expected counts

## Fixed

- On the GPU path, the bifactor GRM E-step returned zero expected counts for
  every general-only item (`specific_map == -1`), in single-group and
  multigroup fits (Issue #2353). `reduce_counts_blk` ran after
  `reduce_counts_gen` and wrote `0.0` over the general-node slots that
  `reduce_counts_gen` had just filled. Zero expected counts prevented
  the affected items' M-step updates.
- Each `counts` slot now has exactly one writer: `reduce_counts_gen` skips
  block items without writing, and `reduce_counts_blk` zeroes only the unused
  tail (`t * qs + h >= qg`) of general-only items. No likelihood, offset,
  dispatch order, or moment code changes.
- `estep_gpu_keeps_general_only_counts_single_and_multigroup` compares GPU
  and CPU expected counts on a two-group fixture with one general-only item.
  Both fixtures also call the reduced GPU kernel directly and require
  successful readback, so CPU fallback cannot satisfy their kernel parity
  assertions. With `MLSIRM_REQUIRE_GPU=1` a missing adapter fails the test.
