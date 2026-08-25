# Polytomous prediction evidence rank admission

## Fixed

- Reject trusted-but-over-rank `theta`, slope, and category-parameter evidence before NumPy materialization or compiled-core discovery.
- Preserve exact 1-D theta/slope and 2-D category-parameter inputs, including exact NumPy row arrays nested in trusted built-in category matrices.
- Keep the existing prediction/evidence cell ceilings and Rust-owned GRM/GPCM probability and expected-score arithmetic unchanged.
