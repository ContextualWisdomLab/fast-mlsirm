# Vectorized NumPy MMLE fallback M-step

## Changed

- The unidimensional 2PL NumPy fallback now updates active item parameters with
  vectorized Newton operations instead of a per-item Python loop. The compiled
  Rust MMLE path remains the preferred production implementation.
- A deterministic missing-data regression fixture preserves discrimination,
  intercept, EAP, and log-likelihood results within explicit floating-point
  tolerances; no bitwise-equivalence or environment-independent speedup claim
  is made.
