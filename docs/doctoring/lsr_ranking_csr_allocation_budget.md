# LSR ranking CSR live allocation budget

## Status

Implemented for Python-side ranking CSR materialization before the Rust LSR/I-LSR handoff. This note documents a bounded resource-accounting correction; it does not change ranking likelihood arithmetic.

## Problem

`_rankings_to_csr` enforced `MAX_RANKING_CSR_BYTES` against logical element counts, then grew `np.empty(..., dtype=uint64)` buffers with a geometric minimum capacity of eight elements and copied each ranking through `np.asarray(list, dtype=uint64)`. On tiny budgets those intermediate capacities and list temporaries could exceed the declared live payload ceiling even when the final sliced arrays fit (NumPy Developers, 2026).

## Decision

- Cap geometric growth by the remaining CSR element budget after reserving start-offset slots.
- Write validated integer item indices directly into the live flat buffer without a list→`uint64` temporary.
- Preserve existing validation, redaction, and Rust ranking solvers.

## Evidence contract

GREEN requires:

- with `MAX_RANKING_CSR_BYTES=32`, materializing one two-item ranking allocates only live payload-sized `uint64` arrays;
- no `np.asarray(list, dtype=uint64)` temporary is created during streaming validation; and
- ordinary ranking validation suites remain green under CI with the compiled core.

## References

NumPy Developers. (2026). *numpy.empty*. NumPy reference. https://numpy.org/doc/stable/reference/generated/numpy.empty.html

Maystre, L., & Grossglauser, M. (2015). Fast and accurate inference of Plackett–Luce models. In *Advances in Neural Information Processing Systems* (pp. 172–180).
