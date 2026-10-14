# Two-tier E-step provenance

## Fixed

- Two-tier GRM fits now take `e_step_n_chunks` and `e_step_n_threads` as
  required caller arguments and record both on the fit result (#2002).
- Bifactor, multigroup bifactor, and two-tier CPU E-steps seed the chunk
  fold from the first partial, so a one-chunk sweep keeps one counts tensor.
