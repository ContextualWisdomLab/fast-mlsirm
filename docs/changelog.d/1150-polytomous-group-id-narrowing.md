# Reject overflowing polytomous DIF labels

## Fixed

- Polytomous DIF group and studied-item label/index vectors now verify signed-64-bit narrowing before compaction or Rust dispatch, preventing unsigned boundary values from wrapping negative and changing group/reference identity.
- Valid non-negative signed-64-bit and sparse/non-contiguous labels remain supported; GRM/GPCM DIF likelihood and statistical arithmetic remain Rust-owned and unchanged.
