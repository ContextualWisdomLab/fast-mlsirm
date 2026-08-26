"""Governed provisional/calibrated parameter-provenance regressions."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric.item_parameter_evidence import (
    ItemParameterStatus,
    ProvisionalParameterMethod,
    build_item_parameter_evidence,
)


def _fingerprint(character: str) -> str:
    """Return one deterministic complete SHA-256-style fixture value."""

    return character * 64


def test_provisional_parameter_evidence_is_explicit_and_content_addressed() -> None:
    """Cold-start parameters cannot masquerade as empirical calibration."""

    evidence = build_item_parameter_evidence(
        item_id="generated_item",
        item_version="1.0.0",
        response_model_id="rasch_model",
        parameter_artifact_fingerprint=_fingerprint("a"),
        status=ItemParameterStatus.PROVISIONAL,
        provisional_method=ProvisionalParameterMethod.RASCH_COMMON_DISCRIMINATION,
        provisional_basis_fingerprint=_fingerprint("b"),
    )

    payload = evidence.to_dict()
    assert payload["status"] == "provisional"
    assert payload["provisional_method"] == "rasch_common_discrimination"
    assert payload["provisional_basis_fingerprint"] == _fingerprint("b")
    assert payload["calibration_evidence_fingerprint"] is None
    assert payload["parameter_artifact_fingerprint"] == _fingerprint("a")
    assert payload["evidence_fingerprint"] == evidence.evidence_fingerprint


def test_calibrated_parameter_evidence_requires_calibration_provenance() -> None:
    """A calibrated claim is bound to exact empirical calibration evidence."""

    evidence = build_item_parameter_evidence(
        item_id="generated_item",
        item_version="1.0.0",
        response_model_id="two_pl_model",
        parameter_artifact_fingerprint=_fingerprint("c"),
        status=ItemParameterStatus.CALIBRATED,
        calibration_evidence_fingerprint=_fingerprint("d"),
    )

    payload = evidence.to_dict()
    assert payload["status"] == "calibrated"
    assert payload["provisional_method"] is None
    assert payload["provisional_basis_fingerprint"] is None
    assert payload["calibration_evidence_fingerprint"] == _fingerprint("d")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        pytest.param(
            {
                "status": ItemParameterStatus.PROVISIONAL,
                "provisional_method": ProvisionalParameterMethod.TEMPLATE_PRIOR,
                "provisional_basis_fingerprint": _fingerprint("e"),
                "calibration_evidence_fingerprint": _fingerprint("f"),
            },
            "provisional parameter evidence must not carry calibration evidence",
            id="provisional-with-calibration",
        ),
        pytest.param(
            {
                "status": ItemParameterStatus.PROVISIONAL,
                "provisional_method": None,
                "provisional_basis_fingerprint": _fingerprint("e"),
            },
            "provisional parameter evidence requires a provisional method",
            id="provisional-without-method",
        ),
        pytest.param(
            {
                "status": ItemParameterStatus.CALIBRATED,
                "provisional_method": ProvisionalParameterMethod.LLTM_PREDICTED,
                "provisional_basis_fingerprint": _fingerprint("e"),
                "calibration_evidence_fingerprint": _fingerprint("f"),
            },
            "calibrated parameter evidence must not carry provisional provenance",
            id="calibrated-with-provisional-provenance",
        ),
        pytest.param(
            {
                "status": ItemParameterStatus.CALIBRATED,
                "calibration_evidence_fingerprint": None,
            },
            "calibrated parameter evidence requires calibration evidence",
            id="calibrated-without-calibration",
        ),
    ],
)
def test_parameter_evidence_rejects_mixed_or_missing_provenance(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Status and provenance cannot be combined into an ambiguous claim."""

    with pytest.raises(ValueError, match=message):
        build_item_parameter_evidence(
            item_id="generated_item",
            item_version="1.0.0",
            response_model_id="rasch_model",
            parameter_artifact_fingerprint=_fingerprint("a"),
            **kwargs,
        )


def test_parameter_evidence_replays_creation_identity_before_serialization() -> None:
    """Frozen-record bypasses cannot relabel provisional parameters as calibrated."""

    evidence = build_item_parameter_evidence(
        item_id="generated_item",
        item_version="1.0.0",
        response_model_id="rasch_model",
        parameter_artifact_fingerprint=_fingerprint("a"),
        status=ItemParameterStatus.PROVISIONAL,
        provisional_method=ProvisionalParameterMethod.CONSTRAINED_PRIOR,
        provisional_basis_fingerprint=_fingerprint("b"),
    )
    object.__setattr__(evidence, "status", ItemParameterStatus.CALIBRATED)

    with pytest.raises(ValueError, match="creation-time identity"):
        evidence.to_dict()


def test_parameter_evidence_rejects_caller_string_subclass_without_callback() -> None:
    """Identifier admission never executes caller-controlled string normalization."""

    calls: list[str] = []

    class HostileString(str):
        def strip(self) -> str:
            calls.append("strip")
            raise AssertionError("caller string callback executed")

    with pytest.raises(ValueError, match="item_id must be a string"):
        build_item_parameter_evidence(
            item_id=HostileString("generated_item"),
            item_version="1.0.0",
            response_model_id="rasch_model",
            parameter_artifact_fingerprint=_fingerprint("a"),
            status=ItemParameterStatus.PROVISIONAL,
            provisional_method=ProvisionalParameterMethod.TEMPLATE_PRIOR,
            provisional_basis_fingerprint=_fingerprint("b"),
        )

    assert calls == []
