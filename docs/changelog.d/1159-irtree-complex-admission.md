# IRTree complex-valued input admission

## Fixed

- Reject complex-valued IRTree response matrices, tree mappings, and node-dimension vectors before any `float64` narrowing can discard imaginary components and change observed categories, mapping branches, or factor assignments.
- Preserve ordinary real-valued categorical responses, `0`/`1`/`NaN` mapping semantics, missingness behavior, and contiguous node-dimension validation without changing any psychometric model formula or estimator arithmetic.
