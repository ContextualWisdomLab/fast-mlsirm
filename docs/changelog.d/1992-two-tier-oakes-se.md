### Added

#### Two-tier / multi-primary Oakes SE and streaming E-step memory (#1992)

- Add `two_tier_oakes_se` (Rust + PyO3 + Python) for confirmatory multi-primary
  / G+W method-factor GRM observed-information SEs via Oakes (1999, eq. 6,
  p. 480), with the same fail-loud non-PD contract as `bifactor_oakes_se`.
- Stream the two-tier E-step / EAP / M-step over the full product Gauss–Hermite
  primary grid without materializing per-item `n_grid * q_specific * n_cat`
  log-prob tables or `n_grid * q_specific * n_primary` node tensors (#1992
  memory), keeping caller-controlled node counts (no silent caps; #1929).
- mirt 1.46.1 Oakes cross-check fixture on the stage-4 dataset
  (`seed=20260917`, `quadpts=15`).
