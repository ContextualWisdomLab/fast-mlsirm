# Dedicated GRM recovery evidence retention

## Changed

- Kept the 500-replication multidimensional Graded Response Model recovery
  study out of pull-request CI and out of the generic 1,800-second ignored-shard
  budget, then published its printed bias, RMSE, convergence, and theta
  correlation lines as a 90-day Actions artifact.
- Withheld checkout credentials from every Statistical Studies job so
  repository-controlled `cargo test` cannot reuse the Actions token.
