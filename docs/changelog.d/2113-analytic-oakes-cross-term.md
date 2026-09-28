# Analytic Oakes cross term for single-group bifactor SEs (#2113)

## Changed

- `bifactor_oakes_se` now computes the Oakes cross term analytically, as the
  per-person posterior covariance of the complete-data score (Oakes, 1999,
  p. 480; the missing information of Louis, 1982). It no longer takes one
  forward difference of the E-step per free parameter. Standard errors can
  differ from earlier releases by the size of that finite-difference error.
- `fd_step` is still a required, validated argument, but the single-group path
  no longer uses it. Removing it will go through the ADR-0028 deprecation
  process.
- This makes the cross term a centred sum of products with no step division,
  which is the precondition for moving it to the f32 GPU kernels later.
