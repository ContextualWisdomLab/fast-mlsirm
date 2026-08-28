"""Creation-seal regressions for semantic-screening evidence."""

from __future__ import annotations

import gc
from pathlib import Path
import runpy
import weakref

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


def test_screening_result_rejects_reused_object_identity_entry() -> None:
    """A registry entry for another live object cannot authorize this result id."""
    original = _result(evaluator_fingerprint=fp("a"))
    other = _result(evaluator_fingerprint=fp("b"))
    record_key = id(original)
    original_entry = screening._RESULT_CREATION_SEALS[record_key]
    screening._RESULT_CREATION_SEALS[record_key] = (
        weakref.ref(other),
        original_entry[1],
    )
    try:
        with pytest.raises(ValueError, match="factory seal"):
            original.to_dict()
    finally:
        screening._RESULT_CREATION_SEALS[record_key] = original_entry


def test_screening_creation_seal_registries_release_discarded_objects() -> None:
    """Package-owned screening seals do not retain discarded checks or results."""
    check = screening.build_semantic_screening_check(
        dimension="answerability",
        status="pass",
        decision_evidence_fingerprint=fp("a"),
    )
    result = _result(evaluator_fingerprint=fp("b"))
    check_key = id(check)
    result_key = id(result)
    check_reference = weakref.ref(check)
    result_reference = weakref.ref(result)

    assert check_key in screening._CHECK_CREATION_SEALS
    assert result_key in screening._RESULT_CREATION_SEALS
    del check
    del result
    gc.collect()

    assert check_reference() is None
    assert result_reference() is None
    assert check_key not in screening._CHECK_CREATION_SEALS
    assert result_key not in screening._RESULT_CREATION_SEALS


def test_screening_replay_rejects_callback_bearing_rebinding_before_equality() -> None:
    """A hostile scalar subclass cannot execute equality during result seal replay."""
    result = _result(evaluator_fingerprint=fp("a"))
    callbacks = 0

    class HostileString(str):
        def __eq__(self, other: object) -> bool:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("caller equality must not run")

    object.__setattr__(result, "evaluator_fingerprint", HostileString(fp("b")))

    with pytest.raises(ValueError, match="factory seal"):
        result.to_dict()

    assert callbacks == 0


def test_screening_result_rejects_equivalent_replacement_check_objects() -> None:
    """Equivalent but newly minted checks cannot replace the creation-time evidence graph."""
    result = _result(evaluator_fingerprint=fp("a"))
    replacement_checks = all_checks()

    object.__setattr__(result, "checks", replacement_checks)

    with pytest.raises(ValueError, match="factory seal"):
        result.to_dict()


def test_screening_check_serialization_uses_verified_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-check rebinding cannot alter serialized screening-check evidence."""
    check = screening.build_semantic_screening_check(
        dimension="answerability",
        status="blocking",
        decision_evidence_fingerprint=fp("a"),
    )
    original_status = check.status
    verify_seal = screening.SemanticScreeningCheck._verify_seal

    def verify_then_rebind(target):
        verified = verify_seal(target)
        object.__setattr__(target, "status", screening.ScreeningStatus.PASS)
        return verified

    monkeypatch.setattr(
        screening.SemanticScreeningCheck,
        "_verify_seal",
        verify_then_rebind,
    )

    payload = check.to_dict()

    assert check.status is screening.ScreeningStatus.PASS
    assert payload["status"] == original_status.value


def test_screening_result_identity_uses_verified_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-check digest rebinding cannot alter the public screening identity."""
    result = _result(evaluator_fingerprint=fp("a"))
    original_fingerprint = result.screening_result_fingerprint
    verify_seal = screening.CandidateScreeningResult._verify_seal

    def verify_then_rebind(target):
        verified = verify_seal(target)
        object.__setattr__(target, "_screening_result_fingerprint", "0" * 64)
        return verified

    monkeypatch.setattr(
        screening.CandidateScreeningResult,
        "_verify_seal",
        verify_then_rebind,
    )

    result_id = result.screening_result_id

    assert vars(result)["_screening_result_fingerprint"] == "0" * 64
    assert result_id == f"screening_result_{original_fingerprint[:32]}"


def test_screening_result_construction_uses_verified_check_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-check dimension rebinding cannot alter result admission structure."""
    item, audit_report = audited_candidate()
    checks = all_checks()
    target = checks[0]
    original_dimension = target.dimension
    verify_seal = screening.SemanticScreeningCheck._verify_seal
    mutated = False

    def verify_then_rebind(check):
        nonlocal mutated
        verified = verify_seal(check)
        if check is target and not mutated:
            object.__setattr__(
                check,
                "dimension",
                screening.ScreeningDimension.AMBIGUITY_MULTIPLE_ANSWER_RISK,
            )
            mutated = True
        return verified

    monkeypatch.setattr(
        screening.SemanticScreeningCheck,
        "_verify_seal",
        verify_then_rebind,
    )

    result = screening.build_candidate_screening_result(
        item,
        audit_report,
        screening_policy_id="semantic_screening_policy",
        screening_policy_version="1.0.0",
        evaluator_kind="hybrid",
        evaluator_fingerprint=fp("a"),
        checks=checks,
    )
    payload = result.to_dict()

    assert target.dimension is screening.ScreeningDimension.AMBIGUITY_MULTIPLE_ANSWER_RISK
    assert payload["checks"][0]["dimension"] == original_dimension.value


def test_valid_screening_creation_seals_preserve_public_identity() -> None:
    """Replay hardening leaves valid screening check/result payloads unchanged."""
    result = _result(evaluator_fingerprint=fp("a"))
    payload = result.to_dict()

    assert payload["screening_result_fingerprint"] == result.screening_result_fingerprint
    assert payload["screening_result_id"] == result.screening_result_id
    assert payload["is_pilot_eligible"] is True
    assert tuple(check.to_dict() for check in result.checks) == tuple(
        payload["checks"]
    )
