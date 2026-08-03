# Precomputed marginal distance hot paths

## Changed

- The marginal estimator's latent interaction term
  `sqrt(eps_distance + ||x - zeta_i||^2)` depends only on the
  `(item, latent-node)` pair, so the probability-table build, the M-step item
  line search, the item gradient, and the tau gradient now reuse a
  precomputed `n_items x n_x` table instead of recomputing the distance and
  `exp(tau)` once per `(context, item, trait-node, latent-node)` quadrature
  cell. Distance-kind results are bit-identical to the scalar reference by
  construction (`a - b == a + (-b)` in IEEE-754, identical accumulation
  order); inner-product models agree to documented floating-point round-off.
- The marginal probability-table fill is sharded over population contexts
  with the crate's coarse fixed-shard `thread::scope` pattern above a
  documented cell floor, with a test-forcible worker seam proving the sharded
  fill bit-identical to the serial fill.
- A same-head benchmark contract
  (`marginal_distance_benchmark_reports_runtime_and_allocation`) emits JSON
  evidence with runtime, output-table and interaction-table bytes, workload
  dimensions, backend label, and the documented 1.05x regression threshold.
  Representative release-mode workload (6 contexts, 60 items, 21 trait
  nodes, 121 latent nodes): scalar reference 37.9 ms, precomputed serial
  20.8 ms, four-worker shard 8.0 ms, interaction-table overhead 58 KB
  against 14.9 MB of output tables (issue #403). No Python estimator or
  fallback arithmetic changed.
