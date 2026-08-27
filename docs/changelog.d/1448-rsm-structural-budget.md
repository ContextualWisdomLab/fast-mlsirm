# RSM response structural and shape budgets

## Fixed

- Bound exact built-in Rating Scale Model response-carrier traversal independently
  of logical numeric cells, so malformed empty-row fan-out cannot consume
  unbounded Python preflight work while keeping the response cell count at zero.
- Preserve the existing 20,000,000-cell RSM evidence envelope and every valid
  non-empty persons-by-items matrix inside it: built-in row plus scalar traversal
  is bounded by twice the logical-cell ceiling before NumPy materialization.
- Replay the established two-dimensional rectangular response contract and the
  minimum-two-item RSM/IRT design contract from inert ndarray shape or exact
  built-in row metadata after resource accounting but before value-wise scans or
  dense float64 marshalling. Small-backed 1-D and one-item broadcast views now
  fail their existing structural diagnostics without first allocating a large
  dense matrix.
- Keep RSM likelihood, marginal-ML EM/ECM, shared-threshold estimation, latent
  integration, scoring, convergence, and uncertainty arithmetic unchanged in the
  Rust core.
