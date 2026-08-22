# IRTree scientific-evidence admission

## Fixed

- Reject complex-valued IRTree response matrices, tree mappings, and node-dimension vectors before any `float64` narrowing can discard imaginary components and change observed categories, mapping branches, or factor assignments.
- Reject arbitrary NumPy array providers, callback-bearing container/scalar subclasses, and object/text storage before package-triggered `__array__` or numeric-conversion callbacks can synthesize or replace IRTree evidence.
- Preserve exact NumPy real-numeric arrays plus exact built-in list/tuple evidence containing package-trusted Python/NumPy real scalars, including ordinary `NaN` missingness, without changing IRTree mapping semantics or psychometric estimator arithmetic.
