# Rasch CML response structural budget

## Fixed

- Bound exact built-in Rasch CML response-tree traversal independently of logical numeric cells, so malformed empty-row fan-out cannot consume unbounded Python preflight work while keeping the response-cell count at zero.
- Preserve the existing 20,000,000-cell response envelope and every valid non-empty persons-by-items matrix inside it by applying a conservative 40,000,000 structural-node ceiling before NumPy materialization.
- Preserve logical-cell error precedence, exact NumPy/list/tuple compatibility, complete 0/1 response semantics, existing dimensionality/minimum-item diagnostics, and Andersen split behavior.
- Keep Rasch conditional likelihood, CML estimation, Andersen likelihood-ratio arithmetic, convergence, and uncertainty unchanged in the Rust numerical core.
