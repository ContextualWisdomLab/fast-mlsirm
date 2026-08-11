# Bifactor advertised-shape metadata bounds

## Decision

Public bifactor scoreability adapters inspect caller-advertised `shape` metadata with bounded look-ahead before NumPy dtype conversion. A two-dimensional matrix shape consumes at most three iterator entries and a one-dimensional uniqueness shape consumes at most two. The extra entry is only a proof of excessive dimensionality; the validator never materializes an unbounded caller iterator merely to learn its length.

Ordinary caller-controlled iteration failures are normalized to the existing package-owned dimensionality `ValueError` and do not echo caller exception text. Process-control signals such as `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` are not swallowed. Accepted arrays and array-like values continue through the existing item/factor/work-budget validation and contiguous `float64` marshalling. All bifactor ECV, PUC, omega, and construct-replicability arithmetic remains Rust-owned.

## Why

Python iterators are not required to be finite. Calling `tuple(shape)` on caller-controlled metadata therefore turns a small structural validation problem into potentially unbounded execution and allocation. Python's iterator protocol also permits arbitrary user code to run from `__iter__()` and `__next__()`, and `StopIteration` is the protocol signal for normal exhaustion. The package only needs a finite prefix to decide whether shape dimensionality is valid, so consuming more input has no product value and expands the denial-of-service and exception-reflection surface.

This is a validation/resource-control decision, not a psychometric or scoreability-method change. The accepted numerical estimand, Rust CPU path, work ceilings, and interpretation boundaries remain unchanged.

## Verification contract

The regression suite must prove that:

- matrix advertised shapes are rejected after at most `2 + 1` requests;
- uniqueness advertised shapes are rejected after at most `1 + 1` requests;
- invalid advertised shapes fail before array conversion/materialization;
- ordinary iterator failures become stable package-owned errors without caller-controlled text;
- process-control signals propagate;
- accepted ordinary NumPy/array-like inputs preserve the existing Rust-backed numerical results; and
- item, factor, and `MAX_BIFACTOR_WORK_UNITS` ceilings remain independently enforced.

## Non-goals

- no generic recursive streaming protocol for arbitrary nested Python containers;
- no scoreability formula, cutoff, model-selection, or interpretation change;
- no GPU path for this bounded non-iterative validation work;
- no change to canonical cross-cutting architecture authority in PR #604/#621.

## References

Python Software Foundation. (2026). *Data model — Python 3.14.6 documentation*. https://docs.python.org/3.14/reference/datamodel.html

Python Software Foundation. (2026). *Built-in exceptions — Python 3.14.6 documentation*. https://docs.python.org/3.14/library/exceptions.html

Python Software Foundation. (2026). *Functional programming HOWTO — Python 3.14.6 documentation*. https://docs.python.org/3.14/howto/functional.html
