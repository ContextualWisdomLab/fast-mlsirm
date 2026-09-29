# Two-tier expected-raw person scores

## Added

- `expected_raw_two_tier_grm(fit, specific_map, q_specific)` returns each
  person's expected raw total on the observed category scale for a fitted
  two-tier GRM, such as the adopted G+4+W emotionality pattern. Primary
  coordinates are fixed at `fit.theta_p_eap`, and each block's specific factor
  is integrated with `q_specific` Gauss-Hermite nodes before the blocks are
  convolved (Lord & Wingersky, 1984; Cai, Yang, & Hansen, 2011). The score is
  the conditional expected raw total at the primary EAP plug-in. It is not the
  mean of the joint primary posterior, so `fit.phi` does not enter it. The
  computation runs in the Rust core.
- `specific_map` accepts `-1` (specific-free) or an integer block index below
  `n_specific`. Fractional, non-finite, and out-of-range entries are rejected in
  fitting, Oakes standard errors, and expected-raw scoring alike, instead of
  being truncated to a different block.
