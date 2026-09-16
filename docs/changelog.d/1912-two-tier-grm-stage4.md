# Single-group polytomous two-tier GRM with reduction over the specific tier (stage 4 of #1912)

## Added

- Add a single-group full-information polytomous two-tier graded response
  fitter (Cai, 2010; Cai, Yang, & Hansen, 2011, eq. 6-7; Gibbons et al.,
  2007, eq. 9/15): caller-supplied confirmatory primary pattern with an
  estimated primary correlation matrix, at most one orthogonal specific
  factor per item, unconstrained slopes, strictly decreasing boundary
  intercepts, Bock-Aitkin EM integrating only `P + 1` dimensions
  (fixed-grid primaries with Phi reweighting, per-block specific sums),
  deterministic multi-start selection, primary-factor EAP scores with
  posterior SDs, and a `fit_two_tier_grm` Python binding. Reduces exactly to
  the stage-1 bifactor GRM at one primary dimension. Validated against
  `mirt::bfactor` with a two-tier specification on a committed fixture
  (slopes/intercepts/correlation/loglik agreement bands with measured values
  reported in the tests).
