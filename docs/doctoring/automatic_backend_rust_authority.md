# Automatic backend resolution must preserve one numerical owner

## Decision under test

`backend="auto"` is a convenience selector for the production Rust/PyO3 numerical implementation. It is not permission to switch to an independent NumPy implementation when the compiled extension is unavailable. A missing compiled core therefore fails closed before psychometric numerical work begins.

The explicit `backend="numpy"` surface is retained in this bounded migration as a reference/parity choice. It is never selected implicitly by automatic production resolution. Rust device selection remains a different axis: a Rust GPU request may fall back to the parity-verified Rust CPU implementation because the numerical owner and formula contract remain Rust-owned.

## Failure boundary

Python's import machinery provides a direct capability probe: `importlib.util.find_spec()` returns `None` when no module specification is found and importing the module is a separate operation. The package uses that boundary to distinguish an unavailable extension from an available compiled module. The PyO3 and maturin primary documentation describe the native extension as the Python-importable compiled module produced and distributed with the package. Those mechanics support a fail-closed contract when the package's required production extension is missing; they do not justify substituting a different numerical implementation.

The package exposes a stable, non-reflective error for an unavailable automatic production backend rather than reflecting local paths, ABI details, environment data, or import exception text into the resolution decision. The message tells the purchaser the next action: install a wheel or editable build that provides `fast_mlsirm._core`, or pass `backend="numpy"` only for the explicit reference/parity path. If discovery succeeds but the native module raises `ImportError` or `OSError` while loading, the package normalizes that loader failure to a stable package-owned `RuntimeError` and retains the original exception as `__cause__` for operator diagnostics; it never interprets an unloadable compiled core as permission to run NumPy arithmetic.

Backend and Rust-device selector strings are also trust-boundary controls. Only exact built-in strings are normalized; arbitrary objects and `str` subclasses are rejected before caller-defined `__str__`, `strip`, or related callbacks and before native-core discovery. Exact built-in values retain the established whitespace/case normalization and allowlist behavior.

## Falsification and acceptance

This decision is falsified if any ordinary `backend="auto"` call can select NumPy because the Rust extension is missing or incompatible. It is also falsified if a purchaser-facing demo, layout, sales-import help, or release-acceptance gate still treats Rust as optional acceleration or treats NumPy as a valid automatic outcome. Acceptance requires tests proving that automatic resolution selects Rust when available, fails closed when absent or unloadable, rejects untrusted selector objects before callbacks/native discovery, and explicit NumPy resolution remains an explicit caller decision. Installed-wheel/package evidence must continue to prove the Rust extension is present in supported production artifacts. Buyer demo copy, repository-layout copy, sales `--check-import` help, and the `fit --backend auto` acceptance check must name the same next action: install `fast_mlsirm._core`, or pass `backend="numpy"` only for the explicit reference/parity path.

## References

Python Software Foundation. (n.d.). *importlib — The implementation of import*. Python 3.14.6 documentation. Retrieved August 10, 2026, from https://docs.python.org/3/library/importlib.html

PyO3 Project. (n.d.). *Building and distribution*. PyO3 user guide. Retrieved August 10, 2026, from https://pyo3.rs/main/building-and-distribution

Maturin Project. (n.d.). *Introduction*. Maturin user guide. Retrieved August 10, 2026, from https://www.maturin.rs/
