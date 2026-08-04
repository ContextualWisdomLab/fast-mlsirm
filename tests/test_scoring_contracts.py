"""Contract tests for automated-scoring assessment specifications."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any
import json
import math

import pytest

import fast_mlsirm.scoring._json as json_module
import fast_mlsirm.scoring.contracts as contracts_module
from fast_mlsirm.rubric.models import ResponseFormat, RubricLevel, RubricSpecification
from fast_mlsirm.scoring import (
    AssessmentSpec,
    ConstructSpec,
    PolicyDocument,
    PolicyKind,
    RubricBinding,
    ScoringContractError,
    build_assessment_spec,
    build_policy_document,
)


def rubric(
    rubric_id: str = "evidence_quality",
    construct_id: str = "evidence_quality",
) -> RubricSpecification:
    """Return one deterministic two-level rubric fixture."""
    return RubricSpecification(
        rubric_id=rubric_id,
        construct_id=construct_id,
        construct_definition="Evidence-conditioned answer quality.",
        response_format=ResponseFormat.ORDINAL_RATING,
        levels=(
            RubricLevel(
                0,
                "not_supported",
                "Not supported.",
                ("claim is unsupported",),
            ),
            RubricLevel(
                1,
                "fully_supported",
                "Fully supported.",
                ("claim is supported",),
            ),
        ),
        task_families=("evidence_review",),
        evidence_requirements=("cite exact evidence",),
    )


def construct(construct_id: str = "evidence_quality") -> ConstructSpec:
    """Return one construct fixture."""
    return ConstructSpec(
        construct_id=construct_id,
        label="Evidence quality",
        definition="Quality conditioned on explicit evidence.",
    )


def policy(kind: PolicyKind, suffix: str | None = None) -> PolicyDocument:
    """Return one deterministic policy fixture."""
    name = suffix or kind.value
    return build_policy_document(
        policy_id=f"{name}_document",
        policy_version="1.0.0",
        policy_kind=kind,
        settings={"policy_mode": "strict", "review_limit": 3},
    )


def policies() -> tuple[PolicyDocument, ...]:
    """Return the complete required policy family."""
    return tuple(policy(kind) for kind in PolicyKind)


def assessment(**overrides: Any) -> AssessmentSpec:
    """Return one governed assessment fixture."""
    values: dict[str, Any] = {
        "assessment_id": "evidence_assessment",
        "assessment_version": "1.0.0",
        "constructs": (construct(),),
        "rubrics": (rubric(),),
        "policy_documents": policies(),
        "metadata": {"deployment_stage": "pilot", "audit_enabled": True},
    }
    values.update(overrides)
    return build_assessment_spec(**values)


class DuplicateKeyMapping(Mapping[str, int]):
    """Mapping fixture whose iterator repeats a key after normalization."""

    def __getitem__(self, key: str) -> int:
        """Return the fixture value."""
        return 1

    def __iter__(self) -> Iterator[str]:
        """Yield the same key twice."""
        yield "duplicate_key"
        yield "duplicate_key"

    def __len__(self) -> int:
        """Return the declared entry count."""
        return 2


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"construct_id": "bad"}, "invalid_construct_id"),
        ({"label": ""}, "invalid_label"),
        ({"definition": ""}, "invalid_definition"),
        ({"schema_version": "2.0"}, "invalid_schema_version"),
    ],
)
def test_construct_rejects_invalid_fields(
    kwargs: dict[str, Any],
    code: str,
) -> None:
    """Construct fields fail with stable redacted codes."""
    values = {
        "construct_id": "evidence_quality",
        "label": "Evidence quality",
        "definition": "A construct definition.",
    }
    values.update(kwargs)
    with pytest.raises(ScoringContractError) as captured:
        ConstructSpec(**values)
    assert captured.value.code == code
    assert captured.value.path.startswith("$")


def test_construct_has_full_and_public_content_identities() -> None:
    """Construct identity is deterministic and uses a 128-bit public handle."""
    value = construct()
    assert len(value.construct_fingerprint) == 64
    assert value.construct_handle == (
        f"construct_spec_{value.construct_fingerprint[:32]}"
    )
    assert value.to_dict()["construct_fingerprint"] == value.construct_fingerprint


def test_policy_is_order_invariant_and_returns_fresh_settings() -> None:
    """Policy identity is canonical across mapping order and never leaks mutability."""
    first = build_policy_document(
        policy_id="engine_policy_document",
        policy_version="1.0.0",
        policy_kind="engine_policy",
        settings={"review_limit": 3, "policy_mode": "strict"},
    )
    second = build_policy_document(
        policy_id="engine_policy_document",
        policy_version="1.0.0",
        policy_kind=PolicyKind.ENGINE,
        settings={"policy_mode": "strict", "review_limit": 3},
    )
    assert first == second
    assert first.policy_fingerprint == second.policy_fingerprint
    assert first.policy_handle.endswith(first.policy_fingerprint[:32])
    copy = first.settings
    copy["review_limit"] = 99
    assert first.settings["review_limit"] == 3


def test_none_settings_are_canonical_empty_policy_settings() -> None:
    """A caller may omit optional policy settings without changing semantics."""
    value = build_policy_document(
        policy_id="engine_policy_document",
        policy_version="1.0.0",
        policy_kind=PolicyKind.ENGINE,
        settings=None,
    )
    assert value.settings == {}


def test_policy_settings_change_the_policy_identity() -> None:
    """A policy-setting change always changes the content identity."""
    first = policy(PolicyKind.ENGINE)
    second = build_policy_document(
        policy_id=first.policy_id,
        policy_version=first.policy_version,
        policy_kind=first.policy_kind,
        settings={"policy_mode": "strict", "review_limit": 4},
    )
    assert first.policy_fingerprint != second.policy_fingerprint


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"policy_id": "bad"}, "invalid_policy_id"),
        ({"policy_version": "01.0.0"}, "invalid_policy_version"),
        ({"policy_kind": "unknown_policy"}, "invalid_policy_kind"),
        ({"settings": []}, "invalid_settings"),
    ],
)
def test_policy_factory_rejects_invalid_contracts(
    kwargs: dict[str, Any],
    code: str,
) -> None:
    """Policy factory rejects invalid identity, kind, version, and settings."""
    values: dict[str, Any] = {
        "policy_id": "engine_policy_document",
        "policy_version": "1.0.0",
        "policy_kind": PolicyKind.ENGINE,
        "settings": {},
    }
    values.update(kwargs)
    with pytest.raises(ScoringContractError) as captured:
        build_policy_document(**values)
    assert captured.value.code == code


def test_factory_seals_governed_policy_rubric_and_assessment_objects() -> None:
    """Direct construction cannot relabel unverified objects as governed artifacts."""
    with pytest.raises(ScoringContractError, match="unverified_policy_document"):
        PolicyDocument(
            "engine_policy_document",
            "1.0.0",
            PolicyKind.ENGINE,
            "{}",
        )
    with pytest.raises(ScoringContractError, match="unverified_rubric_binding"):
        RubricBinding(
            "evidence_quality",
            "1.0.0",
            "0" * 64,
            "evidence_quality",
            ResponseFormat.ORDINAL_RATING,
        )
    with pytest.raises(ScoringContractError, match="unverified_assessment_spec"):
        AssessmentSpec(
            "evidence_assessment",
            "1.0.0",
            (construct(),),
            (),
            policies(),
            "{}",
        )


def test_assessment_is_order_invariant_and_binds_exact_rubrics() -> None:
    """Caller ordering cannot change the governed assessment identity."""
    first_construct = construct("evidence_quality")
    second_construct = construct("response_relevance")
    first_rubric = rubric("evidence_quality", "evidence_quality")
    second_rubric = rubric("response_relevance", "response_relevance")
    first = assessment(
        constructs=(second_construct, first_construct),
        rubrics=(second_rubric, first_rubric),
        policy_documents=tuple(reversed(policies())),
        metadata={"audit_enabled": True, "deployment_stage": "pilot"},
    )
    second = assessment(
        constructs=(first_construct, second_construct),
        rubrics=(first_rubric, second_rubric),
        policy_documents=policies(),
        metadata={"deployment_stage": "pilot", "audit_enabled": True},
    )
    assert first == second
    assert first.rubric_ids == ("evidence_quality", "response_relevance")
    assert first.rubric_fingerprints == (
        first_rubric.fingerprint,
        second_rubric.fingerprint,
    )
    assert first.policy_fingerprints == tuple(
        value.policy_fingerprint for value in policies()
    )
    assert json.loads(first.to_canonical_json()) == first.to_dict()
    assert first.assessment_handle.endswith(first.assessment_fingerprint[:32])
    assert len(first.assessment_fingerprint) == 64
    assert first.policy("engine_policy").policy_kind is PolicyKind.ENGINE


def test_assessment_metadata_is_returned_as_a_fresh_copy() -> None:
    """Assessment metadata cannot be mutated through a returned mapping."""
    value = assessment()
    copy = value.metadata
    copy["audit_enabled"] = False
    assert value.metadata["audit_enabled"] is True


def test_assessment_identity_changes_for_each_governed_input_family() -> None:
    """Construct, rubric, policy, version, and metadata changes alter identity."""
    base = assessment()
    changed = [
        assessment(assessment_version="1.0.1"),
        assessment(
            constructs=(
                ConstructSpec(
                    "evidence_quality",
                    "Evidence",
                    "Changed definition.",
                ),
            )
        ),
        assessment(rubrics=(rubric("alternate_rubric", "evidence_quality"),)),
        assessment(
            policy_documents=(
                build_policy_document(
                    policy_id="engine_policy_document",
                    policy_version="1.0.1",
                    policy_kind=PolicyKind.ENGINE,
                    settings={"policy_mode": "strict", "review_limit": 3},
                ),
                *policies()[1:],
            )
        ),
        assessment(
            metadata={"deployment_stage": "production", "audit_enabled": True}
        ),
    ]
    assert all(
        value.assessment_fingerprint != base.assessment_fingerprint
        for value in changed
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("constructs", "bad", "invalid_constructs"),
        ("constructs", None, "invalid_constructs"),
        ("constructs", (), "invalid_constructs"),
        ("constructs", (object(),), "invalid_construct"),
        ("rubrics", "bad", "invalid_rubrics"),
        ("rubrics", None, "invalid_rubrics"),
        ("rubrics", (), "invalid_rubrics"),
        ("rubrics", (object(),), "invalid_rubric"),
        ("policy_documents", "bad", "invalid_policy_documents"),
        ("policy_documents", None, "invalid_policy_documents"),
        ("policy_documents", (), "invalid_policy_documents"),
        ("policy_documents", (object(),), "invalid_policy_document"),
    ],
)
def test_assessment_factory_rejects_invalid_collections(
    field: str,
    value: Any,
    code: str,
) -> None:
    """Every assessment collection is bounded, typed, and nonempty."""
    values: dict[str, Any] = {
        "assessment_id": "evidence_assessment",
        "assessment_version": "1.0.0",
        "constructs": (construct(),),
        "rubrics": (rubric(),),
        "policy_documents": policies(),
    }
    values[field] = value
    with pytest.raises(ScoringContractError) as captured:
        build_assessment_spec(**values)
    assert captured.value.code == code


def test_assessment_factory_rejects_collection_amplification() -> None:
    """Untrusted iterables cannot exceed construct, rubric, or policy budgets."""
    with pytest.raises(ScoringContractError, match="invalid_constructs"):
        assessment(
            constructs=(construct(f"construct_{index}") for index in range(33))
        )
    with pytest.raises(ScoringContractError, match="invalid_rubrics"):
        assessment(rubrics=(rubric(f"rubric_{index}") for index in range(65)))
    with pytest.raises(ScoringContractError, match="invalid_policy_documents"):
        assessment(
            policy_documents=(
                *policies(),
                policy(PolicyKind.ENGINE, "surplus_engine"),
            )
        )


def test_assessment_rejects_duplicate_and_undeclared_model_contracts() -> None:
    """Duplicate identifiers and undeclared rubric constructs fail closed."""
    item = construct()
    with pytest.raises(ScoringContractError, match="duplicate_construct_id"):
        assessment(constructs=(item, item))
    item_rubric = rubric()
    with pytest.raises(ScoringContractError, match="duplicate_rubric_id"):
        assessment(rubrics=(item_rubric, item_rubric))
    with pytest.raises(ScoringContractError, match="undeclared_rubric_construct"):
        assessment(rubrics=(rubric("other_rubric", "other_construct"),))


def test_assessment_requires_each_policy_kind_once() -> None:
    """Assessments require one and only one policy for each operational family."""
    values = policies()
    with pytest.raises(ScoringContractError, match="missing_policy_kind"):
        assessment(policy_documents=values[:-1])
    with pytest.raises(ScoringContractError, match="duplicate_policy_kind"):
        assessment(
            policy_documents=(
                *values[:-1],
                policy(PolicyKind.ENGINE, "second_engine"),
            )
        )


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        ({"response_text": "secret"}, "sensitive_content_field_forbidden"),
        ({"short": "invalid key"}, "invalid_metadata_key"),
        ({"metric_value": math.inf}, "non_finite_json_number"),
        ({"metric_value": object()}, "unsupported_json_value"),
        ({"long_string": "x" * 2_049}, "json_string_too_long"),
        (
            {f"field_{index}": index for index in range(65)},
            "json_collection_too_large",
        ),
        ({"array_values": list(range(65))}, "json_collection_too_large"),
    ],
)
def test_bounded_json_rejects_unsafe_metadata(
    metadata: Any,
    code: str,
) -> None:
    """Metadata is bounded, finite, descriptive, and free of source/response text."""
    with pytest.raises(ScoringContractError) as captured:
        assessment(metadata=metadata)
    assert captured.value.code == code


def test_bounded_json_rejects_depth_node_duplicate_and_character_budgets() -> None:
    """Every recursive JSON work and output-size budget fails closed."""
    deep: dict[str, Any] = {"leaf_value": 1}
    for index in range(10):
        deep = {f"level_{index}": deep}
    with pytest.raises(ScoringContractError, match="json_depth_exceeded"):
        assessment(metadata=deep)

    nodes = {f"field_{index}": list(range(8)) for index in range(64)}
    with pytest.raises(ScoringContractError, match="json_node_budget_exceeded"):
        assessment(metadata=nodes)

    with pytest.raises(ScoringContractError, match="duplicate_json_key"):
        assessment(metadata=DuplicateKeyMapping())

    oversized = {
        f"field_{index}": ["x" * 2_048] * 8
        for index in range(32)
    }
    with pytest.raises(ScoringContractError, match="invalid_metadata"):
        assessment(metadata=oversized)


def test_json_helpers_cover_null_and_all_scalar_types() -> None:
    """Canonical metadata preserves supported JSON scalar and array values."""
    value = assessment(
        metadata={
            "null_value": None,
            "boolean_value": True,
            "integer_value": 3,
            "float_value": 1.5,
            "string_value": "safe",
            "array_values": [None, False, 2, 2.5, "value"],
        }
    )
    assert value.metadata["array_values"] == [None, False, 2, 2.5, "value"]


def test_internal_decoders_and_error_validation_fail_closed() -> None:
    """Factory storage and structured errors reject malformed internal values."""
    with pytest.raises(ValueError, match="valid JSON"):
        json_module.decode_object_json("{", "payload_json")
    with pytest.raises(ValueError, match="JSON object"):
        json_module.decode_object_json("[]", "payload_json")
    assert json_module.decode_object_json(
        '{"safe_key":1}',
        "payload_json",
    ) == {"safe_key": 1}

    with pytest.raises(ValueError, match="two-or-more-token"):
        ScoringContractError("bad", "$", "message")
    with pytest.raises(ValueError, match="beginning"):
        ScoringContractError("valid_code", "bad", "message")
    with pytest.raises(ValueError, match="must not be empty"):
        ScoringContractError("valid_code", "$", "")


def test_factory_issued_types_recheck_private_invariants() -> None:
    """Private seals still reject corrupted canonical fields and cross-references."""
    policy_token = contracts_module._POLICY_TOKEN
    assessment_token = contracts_module._ASSESSMENT_TOKEN
    binding_token = contracts_module._RUBRIC_BINDING_TOKEN

    with pytest.raises(ScoringContractError, match="noncanonical_policy_settings"):
        PolicyDocument(
            "engine_policy_document",
            "1.0.0",
            PolicyKind.ENGINE,
            '{"review_limit": 3}',
            _policy_token=policy_token,
        )
    with pytest.raises(ValueError, match="valid JSON"):
        PolicyDocument(
            "engine_policy_document",
            "1.0.0",
            PolicyKind.ENGINE,
            "{",
            _policy_token=policy_token,
        )
    with pytest.raises(ValueError, match="JSON object"):
        PolicyDocument(
            "engine_policy_document",
            "1.0.0",
            PolicyKind.ENGINE,
            "[]",
            _policy_token=policy_token,
        )

    with pytest.raises(ScoringContractError, match="invalid_rubric_fingerprint"):
        RubricBinding(
            "evidence_quality",
            "1.0.0",
            "not_a_digest",
            "evidence_quality",
            ResponseFormat.ORDINAL_RATING,
            _binding_token=binding_token,
        )
    with pytest.raises(ScoringContractError, match="invalid_response_format"):
        RubricBinding(
            "evidence_quality",
            "1.0.0",
            "0" * 64,
            "evidence_quality",
            "bad_format",  # type: ignore[arg-type]
            _binding_token=binding_token,
        )

    valid = assessment()
    with pytest.raises(ScoringContractError, match="invalid_constructs"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            (),
            valid.rubric_bindings,
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    with pytest.raises(ScoringContractError, match="duplicate_construct_id"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            (valid.constructs[0], valid.constructs[0]),
            valid.rubric_bindings,
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    alternate = construct("alternate_construct")
    with pytest.raises(ScoringContractError, match="noncanonical_construct_order"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            (valid.constructs[0], alternate),
            valid.rubric_bindings,
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    with pytest.raises(ScoringContractError, match="invalid_rubric_bindings"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            (),
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    with pytest.raises(ScoringContractError, match="duplicate_rubric_id"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            (valid.rubric_bindings[0], valid.rubric_bindings[0]),
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    foreign_binding = RubricBinding(
        "foreign_rubric",
        "1.0.0",
        "0" * 64,
        "foreign_construct",
        ResponseFormat.ORDINAL_RATING,
        _binding_token=binding_token,
    )
    with pytest.raises(ScoringContractError, match="undeclared_rubric_construct"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            (foreign_binding,),
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    later_binding = RubricBinding(
        "zebra_rubric",
        "1.0.0",
        "1" * 64,
        "evidence_quality",
        ResponseFormat.ORDINAL_RATING,
        _binding_token=binding_token,
    )
    with pytest.raises(ScoringContractError, match="noncanonical_rubric_order"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            (later_binding, valid.rubric_bindings[0]),
            valid.policy_documents,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    with pytest.raises(ScoringContractError, match="invalid_policy_documents"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            valid.rubric_bindings,
            valid.policy_documents[:-1],
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    duplicate_policies = (
        *valid.policy_documents[:-1],
        valid.policy_documents[0],
    )
    with pytest.raises(ScoringContractError, match="duplicate_policy_kind"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            valid.rubric_bindings,
            duplicate_policies,
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    with pytest.raises(ScoringContractError, match="noncanonical_policy_order"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            valid.rubric_bindings,
            tuple(reversed(valid.policy_documents)),
            valid.metadata_json,
            _assessment_token=assessment_token,
        )
    with pytest.raises(ScoringContractError, match="noncanonical_assessment_metadata"):
        AssessmentSpec(
            valid.assessment_id,
            valid.assessment_version,
            valid.constructs,
            valid.rubric_bindings,
            valid.policy_documents,
            '{"deployment_stage": "pilot"}',
            _assessment_token=assessment_token,
        )


def test_public_exports_are_explicit() -> None:
    """The scoring namespace exposes only the documented foundational contract API."""
    import fast_mlsirm.scoring as scoring

    for name in (
        "AssessmentSpec",
        "ConstructSpec",
        "PolicyDocument",
        "PolicyKind",
        "RubricBinding",
        "ScoringContractError",
        "build_assessment_spec",
        "build_policy_document",
    ):
        assert getattr(scoring, name) is not None
