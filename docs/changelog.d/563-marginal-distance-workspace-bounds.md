# Bounded marginal latent-distance workspaces

## Fixed

- Replaced the NumPy fallback's item-by-node-by-dimension covariate distance
  broadcast with one bounded squared-norm/matrix-product output that is mutated
  in place.
- Added a private 128 MiB float64 distance-workspace ceiling, checked byte
  products, and pre-allocation gates for pairwise output, live row norms, and
  the intentional item-gradient workspace before latent nodes are built.
- Reused the governed helper in table construction, candidate predictors, the
  tau update, and the covariate update while preserving the Rust production
  backend and public model contracts.

## Security

- Rejected malformed dimensions, Boolean coercion, non-finite or non-float64
  matrices, hidden layout conversions, invalid epsilon, and oversized distance
  workloads with bounded non-reflective diagnostics.
- Added deterministic missing-data and covariate parity tests, a safe
  environment-reporting benchmark, and APA 7 operational doctoring.
