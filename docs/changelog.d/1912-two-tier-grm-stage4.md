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

## Changed

- `q_primary`/`q_specific` validation now resolves the shared arbitrary-`n`
  Gauss-Hermite quadrature (`quadrature::require_gh_rule`, any `n >= 1`,
  #1929) instead of a fixed `SUPPORTED_Q` membership table, matching the
  removal of that table's cap. Add a `#[ignore]`d 121-vs-241 node-count
  numerical-agreement regression (`two_tier_grm_node_agreement.rs`,
  single-primary/single-specific design to keep the primary product grid
  tractable), executed locally (`cargo test --release -- --ignored
  --nocapture`) at both node counts: `q=121` converged in 6 iterations
  (2.20s, final loglik -1246.539916) and `q=241` converged in 6 iterations
  (8.12s, final loglik -1246.539916) — `|loglik diff| = 0.000000`, well
  inside the 5e-3 tolerance.
