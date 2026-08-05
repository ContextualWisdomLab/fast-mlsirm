"""Apply the reviewed PR 546 compatibility fixes before focused validation."""

from __future__ import annotations

from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    """Replace exactly one reviewed text fragment or fail without partial edits."""
    target = Path(path)
    content = target.read_text(encoding="utf-8")
    matches = content.count(old)
    if matches != 1:
        raise SystemExit(f"expected one match in {path}, found {matches}")
    target.write_text(content.replace(old, new), encoding="utf-8")


replace_exact(
    "python/fast_mlsirm/scoring/essay/calibration_reporting.py",
    '''def _category_values(values: Iterable[int]) -> tuple[int, ...]:
''',
    '''def _validate_iteration_trace_length(
    n_iter: int,
    trace_length: int,
    converged: bool,
    *,
    path: str,
) -> None:
    """Require exact Rust trace cardinality without dropping terminal evidence."""
    valid_lengths = {n_iter}
    if not converged:
        valid_lengths.add(n_iter + 1)
    if trace_length not in valid_lengths:
        raise assessment_error(
            "facets_iteration_trace_mismatch",
            path,
            "trace length must equal n_iter, or n_iter + 1 for a "
            "nonconverged terminal evaluation",
        )


def _category_values(values: Iterable[int]) -> tuple[int, ...]:
''',
)
replace_exact(
    "python/fast_mlsirm/scoring/essay/calibration_reporting.py",
    '''    n_iter = _exact_integer(fit.n_iter, "n_iter", minimum=1)
    if n_iter != len(loglik_trace):
        raise assessment_error(
            "facets_iteration_trace_mismatch",
            "$.n_iter",
            "n_iter must equal the number of recorded log-likelihood iterations",
        )
    n_parameters = _exact_integer(fit.n_parameters, "n_parameters", minimum=1)
''',
    '''    n_iter = _exact_integer(fit.n_iter, "n_iter", minimum=1)
    converged = strict_boolean(fit.converged, "converged")
    _validate_iteration_trace_length(
        n_iter,
        len(loglik_trace),
        converged,
        path="$.n_iter",
    )
    n_parameters = _exact_integer(fit.n_parameters, "n_parameters", minimum=1)
''',
)
replace_exact(
    "python/fast_mlsirm/scoring/essay/calibration_reporting.py",
    '''    converged = strict_boolean(fit.converged, "converged")
    design_connected = strict_boolean(design.connected, "design_connected")
''',
    '''    design_connected = strict_boolean(design.connected, "design_connected")
''',
)

replace_exact(
    "python/fast_mlsirm/scoring/essay/calibration_report_html.py",
    '''    n_iter = calibration_reporting._exact_integer(report.n_iter, "n_iter", minimum=1)
    if n_iter != len(loglik_trace):
        raise assessment_error(
            "facets_iteration_trace_mismatch",
            "$.report.n_iter",
            "n_iter must equal the number of log-likelihood iterations",
        )
    n_parameters = calibration_reporting._exact_integer(
''',
    '''    n_iter = calibration_reporting._exact_integer(report.n_iter, "n_iter", minimum=1)
    converged = calibration_reporting.strict_boolean(report.converged, "converged")
    calibration_reporting._validate_iteration_trace_length(
        n_iter,
        len(loglik_trace),
        converged,
        path="$.report.n_iter",
    )
    n_parameters = calibration_reporting._exact_integer(
''',
)
replace_exact(
    "python/fast_mlsirm/scoring/essay/calibration_report_html.py",
    '''    converged = calibration_reporting.strict_boolean(report.converged, "converged")
    design_connected = calibration_reporting.strict_boolean(
''',
    '''    design_connected = calibration_reporting.strict_boolean(
''',
)

replace_exact(
    "tests/test_scoring_enterprise_issue_calibration_reporting.py",
    '''from collections.abc import Iterable

import numpy as np
''',
    '''from collections.abc import Iterable
from pathlib import Path

import numpy as np
''',
)
replace_exact(
    "tests/test_scoring_enterprise_issue_calibration_reporting.py",
    '''from fast_mlsirm.scoring.calibration_reporting import (
    ScoringFacetsCalibrationReport,
    build_scoring_facets_calibration_report,
)
import fast_mlsirm.scoring.enterprise_issue as enterprise
''',
    '''from fast_mlsirm.scoring.calibration_reporting import (
    ScoringFacetsCalibrationReport,
    build_scoring_facets_calibration_report,
)
from fast_mlsirm.scoring.essay import render_essay_facets_calibration_report_html
import fast_mlsirm.scoring.enterprise_issue as enterprise
''',
)
replace_exact(
    "tests/test_scoring_enterprise_issue_calibration_reporting.py",
    '''    assert MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS == 64
''',
    '''    assert MAX_ENTERPRISE_ISSUE_CALIBRATION_REPORTS == 32
''',
)
replace_exact(
    "tests/test_scoring_enterprise_issue_calibration_reporting.py",
    '''def test_actual_rust_fit_produces_one_canonical_report_per_criterion() -> None:
    """The realistic connected fixture crosses the actual Rust-backed fit path."""
    reports = fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_calibration",
        q_theta=7,
        max_iter=8,
        tol=1e-4,
    )

    assert len(reports) == 2
    assert all(type(report) is ScoringFacetsCalibrationReport for report in reports)
    assert all(len(report.respondent_ids) == 2 for report in reports)
    assert all(len(report.task_revision_fingerprints) == 2 for report in reports)
    assert all(len(report.rater_engine_fingerprints) == 2 for report in reports)
    assert all(report.design_connected for report in reports)
    assert all(report.fit_connected for report in reports)
''',
    '''def test_actual_rust_fit_produces_one_canonical_report_per_criterion(
    tmp_path: Path,
) -> None:
    """A capped Rust fit retains and renders its terminal likelihood evidence."""
    reports = fit_enterprise_issue_facets_calibration_reports(
        _connected_executions(),
        report_id_prefix="enterprise_calibration",
        q_theta=7,
        max_iter=1,
        tol=1e-4,
    )

    assert len(reports) == 2
    assert all(type(report) is ScoringFacetsCalibrationReport for report in reports)
    assert all(len(report.respondent_ids) == 2 for report in reports)
    assert all(len(report.task_revision_fingerprints) == 2 for report in reports)
    assert all(len(report.rater_engine_fingerprints) == 2 for report in reports)
    assert all(report.design_connected for report in reports)
    assert all(report.fit_connected for report in reports)
    assert all(report.converged is False for report in reports)
    assert all(len(report.loglik_trace) == report.n_iter + 1 for report in reports)
    assert all(
        "calibration_not_converged" in report.review_trigger_ids
        for report in reports
    )
    for index, report in enumerate(reports):
        output = tmp_path / f"enterprise_calibration_{index}.html"
        render_essay_facets_calibration_report_html(report, output)
        assert output.is_file()
''',
)

replace_exact(
    "docs/automated_essay_facets_calibration_reports.md",
    '''- an iteration count that differs from the trace length;
''',
    '''- a trace length other than `n_iter`, or `n_iter + 1` only when a
  nonconverged Rust fit records its terminal post-update evaluation;
''',
)
replace_exact(
    "docs/automated_essay_facets_calibration_reports.md",
    '''The standalone HTML renderer repeats the numeric, axis, identity, parameter-count, iteration, monotonic-trace, and connectedness checks before serialization. This guards against post-construction mutation or malformed deserialization.
''',
    '''The standalone HTML renderer repeats the numeric, axis, identity, parameter-count, iteration, monotonic-trace, and connectedness checks before serialization. A nonconverged Rust fit may retain one terminal post-update likelihood after its `n_iter` EM iterations; that extra value is evidence, not an additional optimization iteration. This guards against post-construction mutation or malformed deserialization.
''',
)
replace_exact(
    "docs/changelog.d/enterprise-issue-facets-calibration-reports.md",
    '''- Added a realistic connected two-issue, two-task-revision, two-rater-family,
  two-criterion Rust fit, complete orchestration and privacy tests, public
  documentation, and APA 7th scientific and governance traceability.
''',
    '''- Added a realistic connected two-issue, two-task-revision, two-rater-family,
  two-criterion Rust fit, complete orchestration and privacy tests, public
  documentation, and APA 7th scientific and governance traceability.
- Aligned shared report and HTML replay validation with the Rust estimator's
  nonconverged trace contract: `n_iter` optimization iterations may be followed
  by one retained terminal post-update likelihood evaluation.
''',
)
replace_exact(
    "docs/enterprise_issue_facets_calibration_reports.md",
    '''Shared mandatory triggers for nonconvergence or disconnectedness cannot be
suppressed.
''',
    '''Shared mandatory triggers for nonconvergence or disconnectedness cannot be
suppressed. When the Rust estimator reaches its iteration cap, the report retains
the terminal post-update likelihood as one additional trace value without
misreporting it as another optimization iteration.
''',
)
replace_exact(
    "docs/doctoring/enterprise-issue-facets-calibration-reports.md",
    '''The suite also proves execution-order invariance, one-time trigger normalization,
batch rejection of invalid derived report identities before fitting,
reserved-metadata rejection, and absence of raw source and issue text from report
serialization.
''',
    '''The suite also proves execution-order invariance, one-time trigger normalization,
batch rejection of invalid derived report identities before fitting,
reserved-metadata rejection, absence of raw source and issue text from report
serialization, and faithful retention and HTML replay of the Rust estimator's
nonconverged terminal post-update likelihood evaluation.
''',
)
