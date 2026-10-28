# Bootstrap stratum admission (2026-09-27)

Parent: 3e7da8f04ff35944ceff7b15051f817b892b9975, PR #2205.

The real runner converted group IDs to int64 before checking their values.
The negative test with `[0., .5]` reached replicate dispatch as `[0, 0]`.
The stratified sampler allocates an empty vector and only writes positions
whose label is in `0..n_groups-1`; an out-of-range label therefore leaves
an unreadied position. Every declared group must be represented so the
worker fit's inferred group count agrees with the runner's array shapes.

## Opened sources

NumPy Developers, NumPy v2.5 online reference manual, opened 2026-09-27:

- [ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html), Parameters (`casting`) and Examples. Unsafe is the default; the example converts 2.5 to 2. These passages establish why casting alone is not integer-value validation. No newer `same_value` option is required by this fix.
- [numpy.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html), Notes. The operative instruction is “set each element of the array before reading.” This establishes the initialization hazard. Requiring valid, fully represented labels is the existing sampler/worker contract, not a source-prescribed choice of research strata.

Both source URLs and their relevant sections appear in the runner docstring.
Context7 lookup returned monthly quota exceeded; the official manual bodies
were read directly. CodeGraph did not index this Python module; exact Git
source and the complete relevant runner/worker flow were inspected.

## Checks and limits

- Before fix: first new negative test failed with `AssertionError: invalid strata reached replicate dispatch`; 1 failed, 2 deselected.
- After fix: 14 passed in 0.09 seconds in `tests/test_bootstrap_failure_receipts.py` using a modified-module overlay on the existing native candidate environment. Includes invalid fractional, negative, out-of-range, missing, boolean, string and uint64-overflow labels; valid multigroup integer and exact integer-valued float labels preserve strata through the real worker and retain failure receipts.
- `git diff --check` passes. No fresh native package build or immutable release acceptance is established; no study bootstrap estimates are produced.
- Current main's single/multigroup bifactor fit wrappers expose no slope-prior arguments. Accepted AC prior refits and joint DT/E/AC/Z/Y scoring/regression remain separate unfinished requirements. This admission repair does not provide those paths.
