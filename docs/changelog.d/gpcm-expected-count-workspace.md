### Fixed

- Bound the NumPy GPCM parity-reference E-step workspace by accumulating
  Bock–Aitkin expected category counts with weighted `np.bincount`, instead of
  materializing person-by-category boolean and floating-point matrices. The
  compiled Rust core remains the production implementation (#2195).
