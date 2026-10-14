# Two-tier GRM single-primary fit speed

## Fixed

- Single-primary two-tier graded response fits slowed down about tenfold in
  0.11.4, because the E-step recomputed every category log-probability for
  every person. The E-step and the final EAP pass now fill those values once
  per sweep and reuse them across persons. On the reference 500-person,
  6-item, 15 × 11 quadrature fixture, local wall time went from 16.2 s back to
  1.6 s, in line with 0.11.3.
- Multi-primary fits keep the memory-bounded streamed evaluation from 0.11.4.
  Quadrature counts, tolerances, iteration caps, and the model formula are
  unchanged, so fitted values match up to floating-point roundoff.
