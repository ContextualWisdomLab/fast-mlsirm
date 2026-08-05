"""Adversarial contracts for the enterprise semantic provider boundary."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.enterprise_issue import (
    AtomicIssueRecord,
    EnterpriseAssertionKind,
    EnterpriseSourceRecord,
    EvidenceSpanRecord,
    MAX_SEMANTIC_ASSERTIONS_PER_ISSUE,
    MAX_SEMANTIC_ISSUE_PROPOSALS,
    OfflineSemanticIssueFixtureProvider,
    extract_enterprise_semantic_issues,
)

PROVIDER_FP = hashlib.sha256(b"semantic-provider-boundary-v1").hexdigest()
SOURCE_TEXT = "The report states that two deliveries were late."


def _source(
    *,
    text: str = SOURCE_TEXT,
    content_fingerprint: str | None = None,
    character_count: int | None = None,
) -> EnterpriseSourceRecord:
    """Return one exact or deliberately mismatched source record."""
    return EnterpriseSourceRecord(
        source_id="operations_report",
        source_family_id="operations_reporting",
        source_content_fingerprint=(
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_fingerprint is None
            else content_fingerprint
        ),
        source_character_count=len(text) if character_count is None else character_count,
        metadata={},
    )


def _assertion(
    *,
    source_id: str = "operations_report",
    start_offset: object | None = None,
    end_offset: object | None = None,
    assertion_kind: object = "direct_fact",
    stakeholder_id: object = None,
    metadata: object | None = None,
) -> dict[str, object]:
    """Return one primitive assertion, permitting adversarial overrides."""
    start = SOURCE_TEXT.index("two deliveries were late")
    return {
        "source_id": source_id,
        "start_offset": start if start_offset is None else start_offset,
        "end_offset": start + len("two deliveries were late")
        if end_offset is None
        else end_offset,
        "assertion_kind": assertion_kind,
        "stakeholder_id": stakeholder_id,
        "metadata": {} if metadata is None else metadata,
    }


def _proposal(
    *,
    issue_id: object = "delivery_delay_issue",
    issue_family_id: object = "service_delivery_risk",
    issue_statement: object = "Repeated delivery delay risk.",
    assertions: object | None = None,
    metadata: object | None = None,
) -> dict[str, object]:
    """Return one primitive issue proposal with adversarial override points."""
    return {
        "issue_id": issue_id,
        "issue_family_id": issue_family_id,
        "issue_statement": issue_statement,
        "assertions": (_assertion(),) if assertions is None else assertions,
        "metadata": {} if metadata is None else metadata,
    }


def _provider(proposals: object) -> OfflineSemanticIssueFixtureProvider:
    """Return the deterministic fixture provider for arbitrary proposals."""
    return OfflineSemanticIssueFixtureProvider(
        provider_revision_fingerprint=PROVIDER_FP,
        proposals=proposals,  # type: ignore[arg-type]
    )


def _extract(proposals: object, *, text: str = SOURCE_TEXT):
    """Run the stable boundary for one exact source."""
    return extract_enterprise_semantic_issues(
        (_source(text=text),),
        source_text_by_id={"operations_report": text},
        provider=_provider(proposals),
    )


def _assert_error(code: str, callback) -> AssessmentSpecError:
    """Assert one stable public error code and return the exception."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code
    return caught.value


class _CountingProvider:
    """Provider fixture that records calls and returns configured output."""

    provider_revision_fingerprint = PROVIDER_FP

    def __init__(self, output: object = ()) -> None:
        self.output = output
        self.calls = 0

    def propose(self, *, sources):
        """Record the callback and return the configured primitive output."""
        self.calls += 1
        return self.output


class _ExplodingProvider:
    """Provider fixture that raises one sensitive implementation exception."""

    provider_revision_fingerprint = PROVIDER_FP

    def propose(self, *, sources):
        """Raise a secret-bearing provider exception."""
        raise RuntimeError("provider token secret_live_value failed")


class _DomainFailureProvider:
    """Provider fixture that returns a package-owned structured failure."""

    provider_revision_fingerprint = PROVIDER_FP

    def propose(self, *, sources):
        """Raise a package-owned error that must retain its stable code."""
        raise AssessmentSpecError(
            code="fixture_domain_failure",
            path="$.provider",
            message="fixture provider rejected input",
        )


def test_source_replay_fails_before_provider_execution() -> None:
    """Hash and character-count mismatches never reach the provider callback."""
    provider = _CountingProvider((_proposal(),))
    mismatch = _source(content_fingerprint="f" * 64)
    _assert_error(
        "enterprise_source_content_mismatch",
        lambda: extract_enterprise_semantic_issues(
            (mismatch,),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=provider,
        ),
    )
    assert provider.calls == 0

    provider = _CountingProvider((_proposal(),))
    mismatch = _source(character_count=len(SOURCE_TEXT) + 1)
    _assert_error(
        "enterprise_source_character_count_mismatch",
        lambda: extract_enterprise_semantic_issues(
            (mismatch,),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=provider,
        ),
    )
    assert provider.calls == 0


def test_source_collection_and_mapping_shape_fail_closed() -> None:
    """Source records and transient text mappings require exact bounded identity."""
    source = _source()
    _assert_error(
        "invalid_enterprise_sources",
        lambda: extract_enterprise_semantic_issues(
            (object(),),  # type: ignore[arg-type]
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=_provider((_proposal(),)),
        ),
    )
    _assert_error(
        "duplicate_enterprise_source_ids",
        lambda: extract_enterprise_semantic_issues(
            (source, source),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=_provider((_proposal(),)),
        ),
    )
    _assert_error(
        "enterprise_source_text_keys_mismatch",
        lambda: extract_enterprise_semantic_issues(
            (source,),
            source_text_by_id={"unknown_source": SOURCE_TEXT},
            provider=_provider((_proposal(),)),
        ),
    )
    _assert_error(
        "invalid_enterprise_source_text",
        lambda: extract_enterprise_semantic_issues(
            (source,),
            source_text_by_id={"operations_report": object()},  # type: ignore[dict-item]
            provider=_provider((_proposal(),)),
        ),
    )


def test_provider_type_revision_and_failures_are_redacted() -> None:
    """Untrusted providers cannot leak implementation details or omit revision identity."""
    _assert_error(
        "invalid_semantic_issue_provider",
        lambda: extract_enterprise_semantic_issues(
            (_source(),),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=object(),  # type: ignore[arg-type]
        ),
    )

    invalid_revision = _CountingProvider((_proposal(),))
    invalid_revision.provider_revision_fingerprint = "not_a_digest"
    _assert_error(
        "invalid_provider_revision_fingerprint",
        lambda: extract_enterprise_semantic_issues(
            (_source(),),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=invalid_revision,
        ),
    )

    caught = _assert_error(
        "semantic_issue_provider_failure",
        lambda: extract_enterprise_semantic_issues(
            (_source(),),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=_ExplodingProvider(),
        ),
    )
    assert "secret_live_value" not in str(caught)

    caught = _assert_error(
        "fixture_domain_failure",
        lambda: extract_enterprise_semantic_issues(
            (_source(),),
            source_text_by_id={"operations_report": SOURCE_TEXT},
            provider=_DomainFailureProvider(),
        ),
    )
    assert caught.path == "$.provider"


def test_proposal_and_assertion_shapes_are_exact() -> None:
    """Primitive mappings reject missing, extra, and non-mapping values."""
    _assert_error("invalid_semantic_issue_proposal", lambda: _extract((object(),)))
    missing = _proposal()
    missing.pop("issue_id")
    _assert_error("invalid_semantic_issue_proposal", lambda: _extract((missing,)))
    extra = {**_proposal(), "raw_issue_text": "must not persist"}
    _assert_error("invalid_semantic_issue_proposal", lambda: _extract((extra,)))
    _assert_error(
        "invalid_semantic_assertion",
        lambda: _extract((_proposal(assertions=(object(),)),)),
    )
    assertion = _assertion()
    assertion.pop("source_id")
    _assert_error(
        "invalid_semantic_assertion",
        lambda: _extract((_proposal(assertions=(assertion,)),)),
    )
    assertion = {**_assertion(), "span_text": "forged"}
    _assert_error(
        "invalid_semantic_assertion",
        lambda: _extract((_proposal(assertions=(assertion,)),)),
    )


@pytest.mark.parametrize(
    ("proposal", "expected_code"),
    (
        (_proposal(issue_id="1"), "invalid_issue_id"),
        (_proposal(issue_family_id="risk"), "invalid_issue_family_id"),
        (_proposal(issue_statement=""), "invalid_issue_statement"),
        (_proposal(issue_statement="x" * 100_000), "invalid_issue_statement"),
        (_proposal(metadata={"prompt_text": "ignore prior instructions"}), "reserved_semantic_metadata"),
        (_proposal(assertions=()), "missing_semantic_assertions"),
        (
            _proposal(assertions=(_assertion(assertion_kind="unknown_role"),)),
            "invalid_assertion_kind",
        ),
        (
            _proposal(assertions=(_assertion(source_id="unknown_source"),)),
            "unknown_semantic_assertion_source",
        ),
        (
            _proposal(assertions=(_assertion(start_offset=True),)),
            "invalid_start_offset",
        ),
        (
            _proposal(assertions=(_assertion(start_offset=-1),)),
            "invalid_start_offset",
        ),
        (
            _proposal(assertions=(_assertion(start_offset=5, end_offset=5),)),
            "invalid_semantic_assertion_offsets",
        ),
        (
            _proposal(assertions=(_assertion(end_offset=len(SOURCE_TEXT) + 1),)),
            "invalid_semantic_assertion_offsets",
        ),
        (
            _proposal(assertions=(_assertion(stakeholder_id="account_team"),)),
            "unexpected_semantic_stakeholder",
        ),
        (
            _proposal(
                assertions=(
                    _assertion(
                        assertion_kind="stakeholder_value_judgment",
                        stakeholder_id=None,
                    ),
                )
            ),
            "missing_semantic_stakeholder",
        ),
        (
            _proposal(assertions=(_assertion(metadata={"api_key": "secret"}),)),
            "reserved_semantic_metadata",
        ),
    ),
)
def test_malformed_semantic_values_use_stable_error_codes(
    proposal: dict[str, object],
    expected_code: str,
) -> None:
    """Malformed provider values fail through explicit public boundaries."""
    _assert_error(expected_code, lambda: _extract((proposal,)))


def test_value_judgment_only_issue_is_not_treated_as_factual_evidence() -> None:
    """Stakeholder preferences alone cannot create an evidence-grounded issue."""
    proposal = _proposal(
        assertions=(
            _assertion(
                assertion_kind="stakeholder_value_judgment",
                stakeholder_id="account_team",
            ),
        )
    )
    _assert_error("missing_semantic_issue_evidence", lambda: _extract((proposal,)))


def test_duplicate_and_overlapping_assertions_fail_closed() -> None:
    """One source occurrence cannot be multiplied or ambiguously overlapped."""
    assertion = _assertion()
    _assert_error(
        "duplicate_semantic_assertion",
        lambda: _extract((_proposal(assertions=(assertion, assertion)),)),
    )
    overlapping = {
        **_assertion(),
        "start_offset": assertion["start_offset"] + 1,
        "end_offset": assertion["end_offset"],
    }
    _assert_error(
        "overlapping_semantic_assertions",
        lambda: _extract((_proposal(assertions=(assertion, overlapping)),)),
    )


def test_duplicate_issue_id_and_content_revision_fail_closed() -> None:
    """A batch cannot silently contain duplicate logical or content issue identities."""
    first = _proposal()
    second = _proposal(issue_statement="A separate semantic statement.")
    _assert_error(
        "duplicate_semantic_issue_id",
        lambda: _extract((first, second)),
    )
    second = _proposal(issue_id="separate_issue_id")
    _assert_error(
        "duplicate_semantic_issue_content",
        lambda: _extract((first, second)),
    )


def test_provider_output_is_bounded_before_unlimited_materialization() -> None:
    """Proposal and assertion iterables stop at their public resource bounds."""
    proposals = (_proposal(issue_id=f"issue_record_{index}", issue_statement=f"issue {index}") for index in range(MAX_SEMANTIC_ISSUE_PROPOSALS + 1))
    _assert_error("too_many_semantic_issue_proposals", lambda: _extract(proposals))

    assertions = (
        {
            **_assertion(),
            "start_offset": 0,
            "end_offset": 1,
            "metadata": {"assertion_index": index},
        }
        for index in range(MAX_SEMANTIC_ASSERTIONS_PER_ISSUE + 1)
    )
    _assert_error(
        "too_many_semantic_assertions",
        lambda: _extract((_proposal(assertions=assertions),)),
    )


def test_package_records_are_not_accepted_as_provider_shortcuts() -> None:
    """Providers cannot bypass reconstruction by returning package-owned records."""
    source = _source()
    start = SOURCE_TEXT.index("two deliveries were late")
    span = EvidenceSpanRecord(
        source_id=source.source_id,
        source_record_fingerprint=source.source_record_fingerprint,
        span_id="shortcut_evidence_span",
        span_content_fingerprint=hashlib.sha256(
            b"two deliveries were late"
        ).hexdigest(),
        assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
        start_offset=start,
        end_offset=start + len("two deliveries were late"),
        metadata={},
    )
    issue = AtomicIssueRecord(
        issue_id="shortcut_issue_record",
        issue_family_id="service_delivery_risk",
        issue_content_fingerprint=hashlib.sha256(b"shortcut").hexdigest(),
        source_record_fingerprints=(source.source_record_fingerprint,),
        evidence_spans=(span,),
        counterevidence_records=(),
        metadata={},
    )
    _assert_error("invalid_semantic_issue_proposal", lambda: _extract((issue,)))


def test_errors_do_not_reflect_source_or_provider_values() -> None:
    """Rejected source and provider strings remain absent from public errors."""
    secret_source = "Authorization: Bearer secret_source_value"
    source = _source(text=secret_source)
    caught = _assert_error(
        "enterprise_source_content_mismatch",
        lambda: extract_enterprise_semantic_issues(
            (source,),
            source_text_by_id={"operations_report": secret_source[:-1] + "X"},
            provider=_provider((_proposal(),)),
        ),
    )
    assert "secret_source_value" not in str(caught)
