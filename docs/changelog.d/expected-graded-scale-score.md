# Expected graded scale score with integrated nuisance factors

## Added

- `expected_graded_scale_score` returns the expected total score of a graded
  scale at a fixed general-factor value, integrating each item's chosen
  factors over equal-probability normal nodes scaled by the factor variances.
- `compute_expected_graded_item_score` returns one graded item's expected
  category score with caller-supplied nodes and weights per integrated factor.
  Weights that are not a probability measure are rejected.
- `equal_probability_normal_nodes(n)` returns the mid-bin normal quantiles
  `Phi^{-1}((k - 0.5) / n)` with weights `1/n`, exactly symmetric about zero.
  It uses the standard library's AS 241 inverse normal, so no new dependency
  is added.

Item expectations are computed by the compiled Rust core, which also rejects
thresholds that are not strictly decreasing.

Both functions require a keyword-only `max_grid_points` work budget. The exact
Cartesian quadrature size (`prod` of per-axis node counts for an item,
`sum_i n_nodes ** d_i` for a scale) is checked with Python integers before any
grid is built, and over-budget requests raise `ValueError`. Node counts are
never capped; the grid is evaluated in fixed-size blocks, so no grid-sized
array is allocated.
