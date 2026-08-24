# Answer-copying evidence admission

## Fixed

- Reject callback-bearing NumPy array providers, ndarray/container subclasses, and caller-defined numeric subclasses before answer-copying evidence is materialized for Wollack omega, K-index/K1/K2/S1/S2, or GBT.
- Preserve exact NumPy numeric arrays and exact built-in list/tuple evidence containing package-trusted Python/NumPy real scalars, while keeping existing complex, dimensional, finite, index, binary, probability, and relation validation contracts.
- Keep all result-affecting answer-copying statistics and tail/regression arithmetic in the Rust numerical core; this change only hardens Python validation and marshalling.
