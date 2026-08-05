"""Apply deterministic source and test corrections for PR 537."""

from __future__ import annotations

from pathlib import Path


def _replace_once(path: str, old: str, new: str) -> None:
    """Replace one exact block or fail without a partial edit."""
    target = Path(path)
    content = target.read_text(encoding="utf-8")
    if content.count(old) != 1:
        raise SystemExit(f"{path}: expected one replacement")
    target.write_text(content.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    """Apply the current accepted focused corrections."""
    module = "python/fast_mlsirm/scoring/enterprise_issue/semantic_proposals.py"
    _replace_once(
        module,
        "from .._contract_safety import (\n"
        "    descriptive_identifier,\n"
        "    enum_value,\n"
        "    freeze_metadata,\n"
        "    thaw_json_value,\n"
        ")\n"
        "from .._validation import AssessmentSpecError, assessment_error, fingerprint\n",
        "from .._contract_safety import (\n"
        "    descriptive_identifier,\n"
        "    enum_value,\n"
        "    freeze_metadata,\n"
        ")\n"
        "from .._validation import (\n"
        "    AssessmentSpecError,\n"
        "    assessment_error,\n"
        "    fingerprint,\n"
        "    thaw_json_value,\n"
        ")\n",
    )

    tests = "tests/test_scoring_enterprise_semantic_provider_boundary.py"
    _replace_once(
        tests,
        "from fast_mlsirm.scoring.enterprise_issue import (\n"
        "    AtomicIssueRecord,\n"
        "    EnterpriseSourceRecord,\n",
        "from fast_mlsirm.scoring.enterprise_issue import (\n"
        "    AtomicIssueRecord,\n"
        "    EnterpriseAssertionKind,\n"
        "    EnterpriseSourceRecord,\n"
        "    EvidenceSpanRecord,\n",
    )
    _replace_once(
        tests,
        "    source = _source()\n"
        "    issue = AtomicIssueRecord(\n"
        "        issue_id=\"shortcut_issue_record\",\n"
        "        issue_family_id=\"service_delivery_risk\",\n"
        "        issue_content_fingerprint=hashlib.sha256(b\"shortcut\").hexdigest(),\n"
        "        source_record_fingerprints=(source.source_record_fingerprint,),\n"
        "        evidence_spans=(),\n"
        "        counterevidence_records=(),\n"
        "        metadata={},\n"
        "    )\n",
        "    source = _source()\n"
        "    start = SOURCE_TEXT.index(\"two deliveries were late\")\n"
        "    span = EvidenceSpanRecord(\n"
        "        source_id=source.source_id,\n"
        "        source_record_fingerprint=source.source_record_fingerprint,\n"
        "        span_id=\"shortcut_evidence_span\",\n"
        "        span_content_fingerprint=hashlib.sha256(\n"
        "            b\"two deliveries were late\"\n"
        "        ).hexdigest(),\n"
        "        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,\n"
        "        start_offset=start,\n"
        "        end_offset=start + len(\"two deliveries were late\"),\n"
        "        metadata={},\n"
        "    )\n"
        "    issue = AtomicIssueRecord(\n"
        "        issue_id=\"shortcut_issue_record\",\n"
        "        issue_family_id=\"service_delivery_risk\",\n"
        "        issue_content_fingerprint=hashlib.sha256(b\"shortcut\").hexdigest(),\n"
        "        source_record_fingerprints=(source.source_record_fingerprint,),\n"
        "        evidence_spans=(span,),\n"
        "        counterevidence_records=(),\n"
        "        metadata={},\n"
        "    )\n",
    )
    _replace_once(
        tests,
        "            source_text_by_id={\"operations_report\": secret_source + \" changed\"},\n",
        "            source_text_by_id={\"operations_report\": secret_source[:-1] + \"X\"},\n",
    )


if __name__ == "__main__":
    main()
