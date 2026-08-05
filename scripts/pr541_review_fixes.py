"""Apply deterministic CodeRabbit review fixes for PR 541."""

from __future__ import annotations

from pathlib import Path
import re

MAIN_TEST = Path("tests/test_scoring_enterprise_issue_calibration.py")
NESTED_TEST = Path("tests/test_scoring_enterprise_issue_calibration_nested.py")


def _replace_function(text: str, name: str, replacement: str) -> str:
    """Replace one top-level test function by exact name."""
    pattern = re.compile(
        rf"^def {re.escape(name)}\b.*?(?=^def |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise SystemExit(f"expected one function named {name}, found {len(matches)}")
    match = matches[0]
    return text[: match.start()] + replacement.rstrip() + "\n\n\n" + text[match.end() :]


def _main_test() -> None:
    """Move shared fixtures and pin each intended provenance failure path."""
    text = MAIN_TEST.read_text(encoding="utf-8")
    text = text.replace("from pathlib import Path\n", "")
    text = text.replace("import runpy\n", "")

    fixture_start = text.index("_FIXTURES = runpy.run_path(")
    first_test = text.index(
        "def test_public_surface_and_shared_bundle_preserve_exact_enterprise_identity"
    )
    fixture_imports = '''from scoring_enterprise_issue_calibration_fixtures import (
    CRITERION_IDS,
    _assert_error,
    _digest,
    _engine,
    _execution,
    _issue,
    _managed_observation_metadata,
    _rebuild_request,
    _request,
    _result,
    _result_with_replacement,
)


'''
    text = text[:fixture_start] + fixture_imports + text[first_test:]

    privacy_old = '    assert "source text" not in repr(bundle.to_dict()).lower()\n'
    privacy_new = '''    serialized = repr(bundle.to_dict())
    for issue_label in ("alpha", "beta"):
        assert f"source-content:{issue_label}" not in serialized
'''
    if text.count(privacy_old) != 1:
        raise SystemExit("expected one vacuous source-text assertion")
    text = text.replace(privacy_old, privacy_new, 1)

    text = _replace_function(
        text,
        "test_undeclared_observation_evidence_fails_replay",
        '''def test_undeclared_observation_evidence_fails_replay() -> None:
    """Observation evidence must remain a subset of the exact request packet."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    unknown_reference = EvidenceReference(
        source_id="unknown_source",
        span_id="unknown_span",
        content_fingerprint=_digest("unknown-evidence"),
        evidence_role=EvidenceRole.SUPPORTING,
    )
    replacement = build_score_observation(
        observation_id="replacement_unknown_evidence",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(unknown_reference,),
        confidence_metadata=_managed_observation_metadata(
            issue,
            (unknown_reference,),
        ),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].evidence_references",
    )''',
    )
    text = _replace_function(
        text,
        "test_non_abstained_observation_requires_supporting_evidence",
        '''def test_non_abstained_observation_requires_supporting_evidence() -> None:
    """A generic scored observation cannot bypass the enterprise evidence gate."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_support",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=(),
        confidence_metadata=_managed_observation_metadata(issue, ()),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].evidence_references",
    )''',
    )
    text = _replace_function(
        text,
        "test_declared_counterevidence_must_survive_calibration_replay",
        '''def test_declared_counterevidence_must_survive_calibration_replay() -> None:
    """Counterevidence cannot disappear before shared calibration."""
    issue = _issue("counter", include_counterevidence=True)
    engine = _engine("counter")
    request = _request(issue, task_label="counter", engine_label="counter")
    result = _result(
        issue=issue,
        request=request,
        engine=engine,
        scores=(1, 2),
    )
    records = build_enterprise_issue_facets_rating_records(
        issue=issue,
        request=request,
        result=result,
        engine=engine,
    )
    assert len(records) == 2

    supporting = tuple(
        value
        for value in enterprise_issue_evidence_references(issue)
        if value.evidence_role is EvidenceRole.SUPPORTING
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_counter",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=supporting,
        confidence_metadata=_managed_observation_metadata(issue, supporting),
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].evidence_references",
    )''',
    )
    text = _replace_function(
        text,
        "test_observation_managed_metadata_must_replay_exactly",
        '''def test_observation_managed_metadata_must_replay_exactly() -> None:
    """Generic confidence metadata cannot counterfeit enterprise observations."""
    issue, request, _, engine = _execution(
        issue_label="alpha",
        task_label="alpha",
        engine_label="alpha",
        scores=(0, 1),
    )
    replacement = build_score_observation(
        observation_id="replacement_missing_metadata",
        request=request,
        engine=engine,
        criterion_id="claim_support",
        status=ObservationStatus.SCORED,
        score_category=1,
        evidence_references=enterprise_issue_evidence_references(issue),
        confidence_metadata={},
    )
    changed_result = _result_with_replacement(
        issue=issue,
        request=request,
        engine=engine,
        replacement=replacement,
    )
    _assert_error(
        "enterprise_calibration_provenance_mismatch",
        lambda: build_enterprise_issue_facets_rating_records(
            issue=issue,
            request=request,
            result=changed_result,
            engine=engine,
        ),
        path="$.result.observations[0].confidence_metadata",
    )''',
    )
    MAIN_TEST.write_text(text, encoding="utf-8")


def _nested_test() -> None:
    """Import shared fixtures normally instead of executing sibling test modules."""
    text = NESTED_TEST.read_text(encoding="utf-8")
    text = text.replace("from pathlib import Path\n", "")
    text = text.replace("import runpy\n", "")
    fixture_start = text.index("_FIXTURES = runpy.run_path(")
    first_helper = text.index("def _assert_provenance_error")
    fixture_imports = '''from scoring_enterprise_issue_calibration_fixtures import (
    _digest,
    _engine,
    _execution,
    _issue,
    _managed_observation_metadata,
    _rebuild_request,
    _request,
    _result,
    _result_with_replacement,
)


'''
    text = text[:fixture_start] + fixture_imports + text[first_helper:]
    metadata_start = text.index("def _managed_observation_metadata")
    next_test = text.index("def test_mutated_result_observation_fails_with_structured_error")
    text = text[:metadata_start] + text[next_test:]
    NESTED_TEST.write_text(text, encoding="utf-8")


def main() -> None:
    """Apply all deterministic review fixes."""
    _main_test()
    _nested_test()


if __name__ == "__main__":
    main()
