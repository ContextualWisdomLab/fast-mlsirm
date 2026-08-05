# Enterprise facets calibration reports implementation plan

## Objective

Advance issues #544 and #404 with one bounded orchestration function that returns
canonical shared calibration reports from exact governed enterprise executions.

## Task 1 — RED contracts

Create `tests/test_scoring_enterprise_issue_calibration_reporting.py` before the
production implementation. Require:

- exact public exports and constants;
- one report per criterion;
- report ID and package-managed bundle/design/criterion provenance;
- single-pass review-trigger normalization;
- tuning and metadata delegation;
- execution-order invariance;
- a realistic actual Rust fit;
- invalid-prefix, reserved-metadata, and source-privacy failures.

Expected RED: missing reporting module and public symbols.

## Task 2 — Production orchestration

Create `python/fast_mlsirm/scoring/enterprise_issue/reporting.py`.

1. Validate the report prefix with the existing descriptive identifier boundary.
2. Deep-freeze caller metadata and reject package-managed keys.
3. Normalize review triggers once through the shared bounded identifier helper.
4. Call `build_enterprise_issue_facets_calibration_bundle()` exactly once.
5. For each canonical design, add exact bundle/design/criterion metadata.
6. Call `fit_scoring_facets_calibration_report()` exactly once.
7. Return a tuple of existing `ScoringFacetsCalibrationReport` values.
8. Export the function and criterion-count limit from the enterprise package.

Do not add statistical arithmetic, a new report type, a provider SDK, a database
object, or a decision layer.

## Task 3 — Documentation and doctoring

- Add the design specification.
- Extend the enterprise many-facet workflow documentation.
- Add APA 7th doctoring for many-facet estimation, testing standards, ISO/IEC
  42001:2023, and NIST AI 600-1.
- Add an authoritative changelog fragment and render `CHANGELOG.md`.

## Task 4 — Verification

Run on one unchanged head:

```bash
pytest -q tests/test_scoring_enterprise_issue_calibration_reporting.py
pytest -q \
  tests/test_scoring_enterprise_issue_calibration.py \
  tests/test_scoring_enterprise_issue_calibration_nested.py \
  tests/test_scoring_enterprise_issue_calibration_bundle.py \
  tests/test_scoring_enterprise_issue_calibration_reporting.py
coverage run --branch \
  --source=python/fast_mlsirm/scoring/enterprise_issue/reporting.py \
  -m pytest -q tests/test_scoring_enterprise_issue_calibration_reporting.py
coverage report --include='*/reporting.py' --fail-under=100
python scripts/check_docstring_coverage.py
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

Then require the complete Python, Rust/PyO3, package, GPU-no-skip, fuzz, Security
Scan, SAST, current-head review, independent approval, unresolved-thread, and
branch-protection gates.

## Task 5 — Merge and continue

Mark ready only after focused evidence passes. Merge without bypass only when all
exact-head gates pass. Re-enumerate the PR queue and continue issue #404 with
human-anchored recovery/validation or governed decision-support prerequisites.
