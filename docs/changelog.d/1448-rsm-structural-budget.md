# RSM response structural budget

## Fixed

- Bound exact built-in Rating Scale Model response-carrier traversal independently
  of logical numeric cells, so malformed empty-row fan-out cannot consume
  unbounded Python preflight work while keeping the response cell count at zero.
- Preserve the existing 20,000,000-cell RSM evidence envelope and every valid
  non-empty persons-by-items matrix inside it: built-in row plus scalar traversal
  is bounded by twice the logical-cell ceiling before NumPy materialization,
  without changing RSM likelihood, EM/ECM, scoring, convergence, or uncertainty
  arithmetic in the Rust core.
