# Top-1 CSR input bounds

## Status

Implemented for Python-side top-1 choice CSR materialization before the Rust LSR/I-LSR top-1 handoff. This note documents a bounded resource and trust-boundary correction; it does not change top-1 likelihood arithmetic.

## Problem

`_top1_to_csr` previously consumed loser iterables with unbounded `list(losers)`, allowed ordinary outer/inner iterator exceptions to escape with caller payload text, and did not apply the shared ranking CSR byte ceiling used by full-ranking materialization. Unbounded or hostile iterators could therefore stall validation, leak exception text, or allocate beyond the package-owned transport budget (Maystre & Grossglauser, 2015; Python Software Foundation, 2026).

## Decision

- Bound loser consumption to at most `n - 1` items with a package-owned overlong-stream error.
- Normalize ordinary outer/inner iteration failures to stable `ValueError` messages that do not reflect caller exception text; propagate `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`.
- Enforce `MAX_RANKING_CSR_BYTES` against winner + loser + start fixed-width `uint64` counts before allocating handoff arrays.
- Validate into pure-Python integer structures first, then allocate exact-size contiguous `uint64` arrays once.

## Evidence contract

GREEN requires bounded loser consumption, non-reflective outer/inner errors, budget rejection below 32 bytes for a one-loser observation, acceptance at exactly 32 bytes, process-control propagation, and ordinary top-1 suites green under CI with the compiled core.

## References

Maystre, L., & Grossglauser, M. (2015). Fast and accurate inference of Plackett–Luce models. In *Advances in Neural Information Processing Systems* (pp. 172–180).

Python Software Foundation. (2026). *Data model*. Python 3.14 documentation. https://docs.python.org/3.14/reference/datamodel.html
