# JMLE Adam/L-BFGS Rust ownership

## Fixed

- Public JMLE `backend="rust"` routes Adam, L-BFGS, and `adam_lbfgs` sequencing
  through compiled `jmle_optimize` entrypoints so optimizer state updates no longer
  re-implement production arithmetic in Python loops.
