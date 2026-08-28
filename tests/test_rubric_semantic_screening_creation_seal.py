"""Creation-seal regressions for semantic-screening evidence."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_semantic_screening_contract.py"))
)
screening = _FIXTURES["screening"]
audited_candidate = _FIXTURES["audited_candidate"]
all_checks = _FIXTURES["all_checks"]
fp = _FIXTURES["fp"]


def _result(*, evaluator_fingerprint: str):
    """Return one complete factory-created semantic screening result."""
    item, audit_report = audited_candidate()
    return screening.build_candidate_screening_result(
        item,
        audit_report,
        screening_policy_id="semantic_screening_policy",
        screening_policy_version="1.0.0",
        evaluator_kind="hybrid",
        evaluator_fingerprint=evaluator_fingerprint,
        checks=all_checks(),
    )


def test_screening_check_rejects_coherent_post_construction_rebinding() -> None:
    """Rebinding content plus the object-local digest cannot mint new check identity."""
    original = screening.build_semantic_screening_check(
        dimension="answerability",
        status="blocking",
        decision_evidence_fingerprint=fp("a"),
    )
    replacement = screening.build_semantic_screening_check(
        dimension="answerability",
        status="pass",
        decision_evidence_fingerprint=fp("b"),
    )

    object.__setattr__(original, "status", replacement.status)
    object.__setattr__(
        original,
        "decision_evidence_fingerprint",
        replacement.decision_evidence_fingerprint,
    )
    object.__setattr__(
        original,
        "_check_fingerprint",
        vars(replacement)["_check_fingerprint"],
    )

    with pytest.raises(ValueError, match="factory seal"):
        original.to_dict()


def test_screening_result_rejects_coherent_post_construction_rebinding() -> None:
    """Rebinding provenance plus its object-local digest cannot mint new result identity."""
    original = _result(evaluator_fingerprint=fp("a"))
    replacement = _result(evaluator_fingerprint=fp("b"))

    object.__setattr__(
        original,
        "evaluator_fingerprint",
        replacement.evaluator_fingerprint,
    )
    object.__setattr__(
        original,
        "_screening_result_fingerprint",
        vars(replacement)["_screening_result_fingerprint"],
    )

    with pytest.raises(ValueError, match="factory seal"):
        original.to_dict()
    with pytest.raises(ValueError, match="factory seal"):
        _ = original.screening_result_fingerprint
