# Orthogonal GRM report transform: source basis and validation

## Supported contract

This producer implements the report transformation for independent unit-variance
normal factors from Muraki and Carlson (1995, pp. 74–76, equations 23–25; p. 79,
equations 38–39). The normal-ogive transform maps slopes and decreasing boundary
intercepts to standardized loadings, thresholds, communality and uniqueness.
Explicit logistic scaling is an approximation, following the metric convention
in Chalmers (2012, p. 3, equation 1 and ordinal boundaries); p. 20, footnote 3
warns that normal-ogive software can produce slightly different results.

The API accepts one item's full slope vector, boundary intercept vector, input
metric and scale. It does not accept a correlated-factor interpretation, default
a logistic conversion scale, infer factor identification, or carry study maps.
Callers must verify their fitted factor covariance/identification separately.
The normal-ogive metric requires scale=1. The logistic approximation requires
an explicitly chosen positive finite scale and returns approximate=true.

The Rust norm is normalized by the largest slope magnitude or scale before
squaring. This preserves finite results for common very small/large magnitudes
without naive sum-of-squares overflow. Nonfinite parameters, empty vectors,
nondecreasing boundary intercepts, invalid scale and unrepresentable transformed
thresholds are rejected. Ordinary f64 rounding and underflow remain possible;
the result does not certify numerical exactness or fitted-model validity.

## Primary texts actually read

Muraki, E., & Carlson, J. E. (1995). Full-information factor analysis for
polytomous item responses. *Applied Psychological Measurement, 19*(1), 73–90.
https://doi.org/10.1177/014662169501900109

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
https://doi.org/10.18637/jss.v048.i06

The source-check record in late-life commit 4046127 is
`evaluations/runs/20260927-grm-report-transform-sources/`: printed/PDF mapping,
images inspected, Zotero keys and SHA-256 hashes. The source metadata is copied
beside this note. Full copyrighted PDFs are not redistributed by this PR.

## Local checks and their limits

Base: fast-mlsirm main 6dd48140c1a267315c7ad1e63a55a47661449c4a.

```
rustc --edition 2021 --test crates/mlsirm-core/src/grm_report.rs -o /private/tmp/fmls-grm-report-tests-20260927
/private/tmp/fmls-grm-report-tests-20260927
/private/tmp/paper-g7-table5-20260927/.venv/bin/python -m pytest -o pythonpath=/private/tmp/fmls-2183-candidate-20260927/python tests/test_grm_report_transform.py -q
```

Standalone Rust: 3 tests pass (roundtrip/reduction, scale/extreme magnitudes,
malformed/unrepresentable parameters). Python: 3 tests pass in 0.20 seconds
(admission without callbacks, exact forwarding/preservation, native rejection
propagation). Python dispatch is mocked against the existing installed package;
it is not a test of the newly compiled PyO3 function. Initial pytest collection
used this sparse tree and failed because config.py was absent. The override
selects the existing package while loading the new wrapper by its exact file.
An initial object-array test construction invoked its own callback; constructing
the object array by scalar assignment corrected the test setup.

Full crate/PyO3 compilation, installed public API, full-suite tests, hosted gates,
non-author review, merge, immutable release and downstream accepted-fit parity
remain pending. No study numbers are produced. Information condition diagnostics
remain a separate unfinished part of issue #2200; this PR must not close it.

## Installed API acceptance check

`tests/test_grm_report_transform.py::test_installed_public_api_native_roundtrip_and_rejection` calls the public package export without mocking the native function. It checks a synthetic exact normal-ogive case, explicitly scaled logistic approximation metadata, and native metric/boundary rejection. It has no skip fallback.

On 2026-09-27, the existing candidate installation at `/private/tmp/fmls-2183-candidate-20260927/python` fails this check with `ImportError: cannot import name orthogonal_grm_report`, as it predates this API. The three admission/dispatch checks pass separately (3 passed, 1 deselected). This failure does not validate the new binding. Rebuild and install this PR head, then run the entire test file without deselection before accepting the API.
