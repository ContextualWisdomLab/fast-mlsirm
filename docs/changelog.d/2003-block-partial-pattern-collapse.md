# Block partial-pattern collapse for bifactor GRM E-step (#2003)

## Added

- CPU bifactor GRM E-step (single-group and multigroup) and final EAP passes
  (single-group, multigroup, FIPC) evaluate `block_acc` / `log_i` once per
  unique within-block response partial pattern (missingness included), then
  look up by person — Bock and Aitkin (1981, pp. 445, 448) pattern-frequency
  EM applied per bifactor block under Gibbons and Hedeker (1992, pp. 423,
  425) / Gibbons et al. (2007, pp. 5, 7–8) between-block conditional
  independence. Provenance fields on `BifactorGrmResult` record measured
  before/after unique counts (ADR-0028). Source-check:
  `docs/papers/2003-block-partial-pattern-source-check.md`. Two-tier GRM
  streaming E-step (#1992) is deferred as a follow-up with the same paper
  basis.
