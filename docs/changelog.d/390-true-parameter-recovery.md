# Rust-only literature true-parameter recovery gate

## Added

- A Rust-only true-parameter recovery experiment for a bounded representative
  Kang and Jeon (2025) MLS2PLM simulation cell (`P = 500`), tracing the
  simple-structure equation, sign convention, identification handling,
  recovery metrics, and citations.
- Orientation-invariant latent-map recovery metrics covering item parameters,
  person traits, person and item interaction positions, and distance weights.
- Scheduled, manual-dispatch, and release-tag statistical-study workflows that
  execute exhaustive ignored Rust studies in exact-name-validated shards while
  pull-request CI retains bounded CPU/GPU sentinels.
- A source-backed finite-Monte-Carlo convergence floor
  (`p0 - 2 * sqrt(p0 * (1 - p0) / R)`) for the 500-replication higher-order
  DINA recovery study.

## Changed

- The duplicate NumPy-only recovery experiment is removed; the Rust core is
  the single evidence path for literature recovery gates.
- The historical `cdm::tests::mc_ho_recovery_500` study is removed at the
  source level. Its generating design, fixed seeds, and RMSE, bias, and
  agreement thresholds are preserved verbatim by the reviewed
  `higher_order_dina_recovery_respects_monte_carlo_tolerance` integration
  study, which gates convergence on the documented two-standard-error binomial
  floor instead of an exact finite-sample proportion.

## Fixed

- Ignored-test shard discovery rejects stale skip declarations, duplicate
  skips, ambiguous final-component exclusions, and silently empty shards.
- Explicit-GPU parity evidence fails closed when the Vulkan adapter is
  unavailable instead of silently skipping.
