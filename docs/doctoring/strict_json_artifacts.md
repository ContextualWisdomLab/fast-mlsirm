# Strict finite JSON artifact serialization

## Decision

Package-owned JSON artifacts must use the RFC 8259 number grammar. `NaN`, positive infinity, and negative infinity are not valid JSON numbers and therefore must never be emitted by governed `fast-mlsirm` artifact writers.

Python's standard `json` encoder deliberately defaults `allow_nan=True` for JavaScript compatibility, which emits `NaN`, `Infinity`, and `-Infinity`. `fast-mlsirm` overrides that compatibility default with `allow_nan=False` at one shared serialization boundary so a non-finite value fails before the destination JSON artifact is atomically replaced.

The failure surface is intentionally non-reflective: non-finite numeric failures use the package-owned message `artifact contains a non-finite JSON numeric value`, while other JSON serialization failures use a separate bounded package-owned error. The rejected payload, raw source content, local path, provider output, and other caller-controlled material are not interpolated into either error.

## Scope

The shared strict serializer governs JSON emitted by:

- `save_simulation()` configuration and manifest artifacts;
- `save_fit_result()` summary artifacts;
- `save_fit_diagnostics()` diagnostic artifacts; and
- `save_dimensionality_diagnostics()` dimensionality-search artifacts.

NumPy `.npy`/`.npz` numerical artifacts are a separate binary format and are not converted to JSON merely to satisfy this decision. This change does not alter estimators, fit statistics, parameter values, or the package's Rust/PyO3 numerical ownership.

## Verification

The regression suite tests nested `NaN`, `Infinity`, and `-Infinity` values at the governed public writer boundaries. Each value must fail without publishing a new target or replacing a pre-existing JSON artifact. A circular-reference regression separately proves that non-numeric serialization failures are not mislabeled as non-finite numeric failures. Finite IEEE-754 binary64 extreme values remain serializable and must round-trip through a strict JSON parser. Existing atomic-write semantics remain the publication boundary.

The fail-first head `90f3ccfb23dafea528d9b7c43d2cc57457fe7081` reached the public dimensionality-artifact writer and completed the full Python suite with exactly one intended failure because the old writer did not raise on `NaN`. The first GREEN implementation head `d8e22e3ecd793d40273244f9cb6e007d7f0a57ba` then completed the full Python suite with 2,949 passed and 2 skipped; the strengthened replacement must recreate all acceptance evidence from the current protected-main lineage.

## References

Bray, T. (Ed.). (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259; STD 90). RFC Editor. https://www.rfc-editor.org/rfc/rfc8259

Python Software Foundation. (2026). *json — JSON encoder and decoder* (Python 3.14.6 documentation). https://docs.python.org/3.14/library/json.html
