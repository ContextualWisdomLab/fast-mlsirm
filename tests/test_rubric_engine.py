"""Behavioral contract for the offline rubric-to-item blueprint compiler."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import re

import pytest

from fast_mlsirm.rubric_engine import (
    GENERATED_ITEM_SCHEMA_DRAFT,
    MAX_COMPILED_BLUEPRINTS,
    SUPPORTED_RUBRIC_SCHEMA_VERSION,
    BlueprintPayload,
    CompiledBlueprint,
    DifficultyBand,
    EvidenceMode,
    GenerationContract,
    GenerationContractPayload,
    RubricCriterion,
    RubricLevel,
    RubricSpecification,
    TaskFamily,
    build_generation_contract,
    canonical_json,
    compile_item_blueprints,
    generated_item_output_schema,
    sha256_fingerprint,
)


def _level(score: int, suffix: str) -> RubricLevel:
    """Build one valid ordinal level for tests."""
    return RubricLevel(
        score=score,
        label=f"Level {suffix}",
        description=f"Description {suffix}.",
        observable_indicators=(f"Observable indicator {suffix}.",),
    )


def _criterion(identifier: str = "business_materiality") -> RubricCriterion:
    """Build one valid criterion for tests."""
    return RubricCriterion(
        criterion_id=identifier,
        title="Business materiality",
        construct="decision_consequence",
        description="Expected consequence if the issue is realized.",
        levels=(_level(0, "zero"), _level(1, "one"), _level(2, "two")),
        prohibited_patterns=("emotion_only_rationale",),
    )


def _task(identifier: str = "evidence_issue_analysis") -> TaskFamily:
    """Build one valid task family for tests."""
    return TaskFamily(
        task_family_id=identifier,
        description="Analyze one atomic issue against supplied evidence.",
        expected_response_format="structured_json_record",
        required_reasoning=("identify_support", "identify_counterevidence"),
    )


def _evidence(identifier: str = "bounded_source_packet") -> EvidenceMode:
    """Build one valid evidence mode for tests."""
    return EvidenceMode(
        evidence_mode_id=identifier,
        description="Use only the supplied source packet.",
        evidence_requirement="Every material claim names a supplied source.",
        minimum_sources=1,
        maximum_sources=3,
        allowed_source_types=("enterprise_report", "customer_record"),
    )


def _difficulty(
    identifier: str = "direct_evidence_band", order_index: int = 0
) -> DifficultyBand:
    """Build one valid difficulty band for tests."""
    return DifficultyBand(
        difficulty_band_id=identifier,
        order_index=order_index,
        description="Direct, explicit evidence.",
        constraints=("single_issue", "explicit_consequence"),
    )


def _valid_specification(**overrides) -> RubricSpecification:
    """Build the smallest valid rubric specification."""
    values = {
        "schema_version": "1.0.0",
        "rubric_id": "customer_issue_priority",
        "rubric_version": "1.0.0",
        "title": "Customer issue priority",
        "purpose": "Measure evidence-conditioned business issue priority.",
        "locale": "ko-KR",
        "response_format": "structured_json_record",
        "criteria": (_criterion(),),
        "task_families": (_task(),),
        "evidence_modes": (_evidence(),),
        "difficulty_bands": (_difficulty(),),
        "prohibited_patterns": ("invented_evidence",),
        "generation_rules": ("preserve_counterevidence",),
    }
    values.update(overrides)
    return RubricSpecification(**values)


def _multi_axis_specification(*, reverse_axes: bool = False) -> RubricSpecification:
    """Build a two-by-two-by-two-by-two specification."""
    criteria = (_criterion("business_materiality"), _criterion("action_urgency"))
    tasks = (_task("evidence_issue_analysis"), _task("comparative_issue_ranking"))
    evidence = (
        _evidence("bounded_source_packet"),
        _evidence("cross_source_packet"),
    )
    difficulty = (
        _difficulty("direct_evidence_band", 0),
        _difficulty("conflicting_evidence_band", 1),
    )
    if reverse_axes:
        criteria = tuple(reversed(criteria))
        tasks = tuple(reversed(tasks))
        evidence = tuple(reversed(evidence))
        difficulty = tuple(reversed(difficulty))
    return _valid_specification(
        criteria=criteria,
        task_families=tasks,
        evidence_modes=evidence,
        difficulty_bands=difficulty,
    )


def test_supported_public_constants_are_pinned():
    """Schema and allocation contracts remain explicit at the public boundary."""
    assert SUPPORTED_RUBRIC_SCHEMA_VERSION == "1.0.0"
    assert GENERATED_ITEM_SCHEMA_DRAFT == "https://json-schema.org/draft/2020-12/schema"
    assert MAX_COMPILED_BLUEPRINTS == 100_000


@pytest.mark.parametrize(
    "identifier",
    ["single", "Upper_Case", "1_invalid_name", "invalid-name", "bad__name", "_bad_name"],
)
def test_domain_identifiers_require_two_or_more_snake_case_tokens(identifier):
    """Every persisted domain identifier follows the repository naming rule."""
    with pytest.raises(ValueError, match="criterion_id.*snake_case"):
        _criterion(identifier)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "v1", "schema_version.*semantic"),
        ("schema_version", "1.1.0", "supported schema_version"),
        ("rubric_version", "one", "rubric_version.*semantic"),
        ("rubric_id", "single", "rubric_id.*snake_case"),
        ("title", "  ", "title.*non-empty"),
        ("purpose", "", "purpose.*non-empty"),
        ("locale", "not a locale", "locale"),
        ("response_format", "single", "response_format.*snake_case"),
    ],
)
def test_rubric_metadata_is_strictly_validated(field, value, message):
    """Version, identity, locale, and descriptive metadata fail closed."""
    with pytest.raises(ValueError, match=message):
        _valid_specification(**{field: value})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score": True}, "score.*integer"),
        ({"score": -1}, "score.*non-negative"),
        ({"label": ""}, "label.*non-empty"),
        ({"description": "  "}, "description.*non-empty"),
        ({"observable_indicators": ()}, "observable_indicators.*at least one"),
        ({"observable_indicators": ("",)}, "observable_indicators.*non-empty"),
        (
            {"observable_indicators": ("same", "same")},
            "observable_indicators.*unique",
        ),
    ],
)
def test_rubric_level_validation(kwargs, message):
    """Ordinal levels contain auditable non-empty evidence indicators."""
    values = {
        "score": 0,
        "label": "No consequence",
        "description": "No consequence is supported.",
        "observable_indicators": ("No decision consequence is supported.",),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RubricLevel(**values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"criterion_id": "single"}, "criterion_id.*snake_case"),
        ({"title": ""}, "title.*non-empty"),
        ({"construct": "single"}, "construct.*snake_case"),
        ({"description": ""}, "description.*non-empty"),
        ({"levels": ()}, "levels.*at least one"),
        ({"prohibited_patterns": ("",)}, "prohibited_patterns.*non-empty"),
        (
            {"prohibited_patterns": ("duplicate_pattern", "duplicate_pattern")},
            "prohibited_patterns.*unique",
        ),
    ],
)
def test_criterion_metadata_validation(kwargs, message):
    """Criterion definitions reject incomplete or ambiguous metadata."""
    values = {
        "criterion_id": "business_materiality",
        "title": "Business materiality",
        "construct": "decision_consequence",
        "description": "Expected consequence.",
        "levels": (_level(0, "zero"),),
        "prohibited_patterns": (),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RubricCriterion(**values)


@pytest.mark.parametrize(
    "levels",
    [
        (_level(0, "zero"), _level(0, "duplicate")),
        (_level(0, "zero"), _level(2, "gap")),
        (_level(1, "one"), _level(2, "two")),
    ],
)
def test_criterion_scores_start_at_zero_and_are_unique_contiguous(levels):
    """Ordinal score levels form the exact integer range zero through maximum."""
    with pytest.raises(ValueError, match="levels.*contiguous.*zero"):
        replace(_criterion(), levels=levels)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (_task, {"identifier": "single"}, "task_family_id.*snake_case"),
        (_evidence, {"identifier": "single"}, "evidence_mode_id.*snake_case"),
        (
            _difficulty,
            {"identifier": "single", "order_index": 0},
            "difficulty_band_id.*snake_case",
        ),
    ],
)
def test_axis_identifiers_are_strict(factory, kwargs, message):
    """Task, evidence, and difficulty identifiers use the same naming contract."""
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"description": ""}, "description.*non-empty"),
        ({"expected_response_format": "single"}, "expected_response_format.*snake_case"),
        ({"required_reasoning": ()}, "required_reasoning.*at least one"),
        ({"required_reasoning": ("",)}, "required_reasoning.*non-empty"),
        (
            {"required_reasoning": ("identify_support", "identify_support")},
            "required_reasoning.*unique",
        ),
    ],
)
def test_task_family_validation(kwargs, message):
    """Task families carry explicit response and reasoning requirements."""
    values = {
        "task_family_id": "evidence_issue_analysis",
        "description": "Analyze evidence.",
        "expected_response_format": "structured_json_record",
        "required_reasoning": ("identify_support",),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TaskFamily(**values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"description": ""}, "description.*non-empty"),
        ({"evidence_requirement": ""}, "evidence_requirement.*non-empty"),
        ({"minimum_sources": True}, "minimum_sources.*integer"),
        ({"minimum_sources": -1}, "minimum_sources.*non-negative"),
        ({"maximum_sources": False}, "maximum_sources.*integer"),
        ({"maximum_sources": -1}, "maximum_sources.*non-negative"),
        (
            {"minimum_sources": 3, "maximum_sources": 2},
            "maximum_sources.*minimum_sources",
        ),
        ({"allowed_source_types": ()}, "allowed_source_types.*at least one"),
        ({"allowed_source_types": ("single",)}, "allowed_source_types.*snake_case"),
        (
            {"allowed_source_types": ("customer_record", "customer_record")},
            "allowed_source_types.*unique",
        ),
    ],
)
def test_evidence_mode_validation(kwargs, message):
    """Evidence modes bound source use and make allowed evidence explicit."""
    values = {
        "evidence_mode_id": "bounded_source_packet",
        "description": "Use supplied evidence.",
        "evidence_requirement": "Cite supplied evidence.",
        "minimum_sources": 1,
        "maximum_sources": 2,
        "allowed_source_types": ("customer_record",),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        EvidenceMode(**values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order_index": True}, "order_index.*integer"),
        ({"order_index": -1}, "order_index.*non-negative"),
        ({"description": ""}, "description.*non-empty"),
        ({"constraints": ()}, "constraints.*at least one"),
        ({"constraints": ("single",)}, "constraints.*snake_case"),
        (
            {"constraints": ("single_issue", "single_issue")},
            "constraints.*unique",
        ),
    ],
)
def test_difficulty_band_validation(kwargs, message):
    """Difficulty bands have a stable order and explicit generation constraints."""
    values = {
        "difficulty_band_id": "direct_evidence_band",
        "order_index": 0,
        "description": "Direct evidence.",
        "constraints": ("single_issue",),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        DifficultyBand(**values)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("criteria", "criteria.*at least one"),
        ("task_families", "task_families.*at least one"),
        ("evidence_modes", "evidence_modes.*at least one"),
        ("difficulty_bands", "difficulty_bands.*at least one"),
    ],
)
def test_specification_requires_every_compilation_axis(field, message):
    """The compiler never infers an absent evidence-centered design axis."""
    with pytest.raises(ValueError, match=message):
        _valid_specification(**{field: ()})


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("criteria", (_criterion(), _criterion()), "criteria.*duplicate"),
        ("task_families", (_task(), _task()), "task_families.*duplicate"),
        ("evidence_modes", (_evidence(), _evidence()), "evidence_modes.*duplicate"),
        (
            "difficulty_bands",
            (_difficulty(), _difficulty()),
            "difficulty_bands.*duplicate",
        ),
    ],
)
def test_specification_rejects_duplicate_axis_identifiers(field, values, message):
    """Duplicate axis members cannot create duplicate blueprint identities."""
    with pytest.raises(ValueError, match=message):
        _valid_specification(**{field: values})


@pytest.mark.parametrize(
    "bands",
    [
        (_difficulty("direct_evidence_band", 0), _difficulty("conflicting_evidence_band", 0)),
        (_difficulty("direct_evidence_band", 0), _difficulty("conflicting_evidence_band", 2)),
        (_difficulty("direct_evidence_band", 1),),
    ],
)
def test_difficulty_indices_are_unique_zero_based_and_contiguous(bands):
    """Difficulty ordering cannot depend on a sparse or duplicated index."""
    with pytest.raises(ValueError, match="difficulty_bands.*contiguous.*zero"):
        _valid_specification(difficulty_bands=bands)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prohibited_patterns", ("",), "prohibited_patterns.*non-empty"),
        (
            "prohibited_patterns",
            ("invented_evidence", "invented_evidence"),
            "prohibited_patterns.*unique",
        ),
        ("generation_rules", ("",), "generation_rules.*non-empty"),
        (
            "generation_rules",
            ("preserve_counterevidence", "preserve_counterevidence"),
            "generation_rules.*unique",
        ),
    ],
)
def test_specification_rule_lists_are_nonempty_and_unique(field, value, message):
    """Global generation controls cannot contain blank or duplicate rules."""
    with pytest.raises(ValueError, match=message):
        _valid_specification(**{field: value})


def test_schema_objects_are_frozen_and_normalized():
    """Validated authoring inputs cannot drift after fingerprinting."""
    specification = _multi_axis_specification(reverse_axes=True)
    assert [item.criterion_id for item in specification.criteria] == [
        "action_urgency",
        "business_materiality",
    ]
    assert [item.task_family_id for item in specification.task_families] == [
        "comparative_issue_ranking",
        "evidence_issue_analysis",
    ]
    assert [item.order_index for item in specification.difficulty_bands] == [0, 1]
    with pytest.raises(FrozenInstanceError):
        specification.title = "Changed"  # type: ignore[misc]


def test_canonical_json_is_sorted_compact_and_utf8():
    """Canonical text is stable and retains non-ASCII content."""
    value = {"z_key": 2, "a_key": "한글", "nested_value": [True, None, 3]}
    assert canonical_json(value) == (
        '{"a_key":"한글","nested_value":[true,null,3],"z_key":2}'
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"bad_value": 0.5}, "floating-point"),
        (b"bytes", "bytes"),
        ({"bad_set"}, "set"),
        ({1: "not_string_key"}, "string keys"),
        (object(), "unsupported canonical type"),
    ],
)
def test_canonical_json_rejects_ambiguous_or_language_specific_types(value, message):
    """Contract identity never depends on lossy or implementation-specific coercion."""
    with pytest.raises(ValueError, match=message):
        canonical_json(value)


def test_canonical_json_serializes_frozen_schema_objects():
    """Schema objects use field names and normalized tuple order in canonical JSON."""
    text = canonical_json(_valid_specification())
    decoded = json.loads(text)
    assert decoded["rubric_id"] == "customer_issue_priority"
    assert decoded["criteria"][0]["levels"][2]["score"] == 2


def test_fingerprints_are_sha256_and_semantically_stable():
    """Canonical ordering gives equivalent authoring inputs the same identity."""
    left = sha256_fingerprint(_multi_axis_specification(reverse_axes=False))
    right = sha256_fingerprint(_multi_axis_specification(reverse_axes=True))
    assert left == right
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", left)


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "A materially different purpose."},
        {"locale": "en-US"},
        {"response_format": "auditable_json_record"},
        {"prohibited_patterns": ("invented_evidence", "missing_counterevidence")},
        {"generation_rules": ("preserve_counterevidence", "separate_fact_inference")},
    ],
)
def test_fingerprints_change_when_material_rubric_constraints_change(change):
    """Every material authoring constraint participates in content identity."""
    original = _valid_specification()
    changed = _valid_specification(**change)
    assert sha256_fingerprint(original) != sha256_fingerprint(changed)


@pytest.mark.parametrize("maximum", [True, 0, -1, MAX_COMPILED_BLUEPRINTS + 1, 1.5])
def test_compiler_validates_caller_controlled_product_limit(maximum):
    """Invalid or unsafe product limits are rejected before compilation."""
    with pytest.raises(ValueError, match="max_blueprints"):
        compile_item_blueprints(_valid_specification(), max_blueprints=maximum)


def test_compiler_requires_a_rubric_specification():
    """Arbitrary objects cannot enter the compiler through duck typing."""
    with pytest.raises(ValueError, match="RubricSpecification"):
        compile_item_blueprints({"rubric_id": "bad_input"}, max_blueprints=1)


def test_compiler_emits_complete_deterministic_cartesian_product():
    """All configured design combinations are emitted exactly once."""
    specification = _multi_axis_specification(reverse_axes=True)
    blueprints = compile_item_blueprints(specification, max_blueprints=16)
    repeated = compile_item_blueprints(specification, max_blueprints=16)
    assert len(blueprints) == 16
    assert blueprints == repeated
    assert len({item.blueprint_id for item in blueprints}) == 16
    assert all(item.blueprint_id.startswith("item_blueprint_") for item in blueprints)
    assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", item.blueprint_fingerprint) for item in blueprints)
    first = blueprints[0].payload
    last = blueprints[-1].payload
    assert (
        first.criterion_id,
        first.task_family_id,
        first.evidence_mode_id,
        first.difficulty_band_id,
    ) == (
        "action_urgency",
        "comparative_issue_ranking",
        "bounded_source_packet",
        "direct_evidence_band",
    )
    assert (
        last.criterion_id,
        last.task_family_id,
        last.evidence_mode_id,
        last.difficulty_band_id,
    ) == (
        "business_materiality",
        "evidence_issue_analysis",
        "cross_source_packet",
        "conflicting_evidence_band",
    )


def test_compiler_fails_instead_of_silently_truncating():
    """A product above the caller bound is reported before partial output exists."""
    with pytest.raises(ValueError, match=r"product size 16 exceeds max_blueprints 15"):
        compile_item_blueprints(_multi_axis_specification(), max_blueprints=15)


def test_compiled_blueprint_contains_complete_traceability():
    """One blueprint carries every constraint required by a downstream adapter."""
    specification = _valid_specification()
    blueprint = compile_item_blueprints(specification, max_blueprints=1)[0]
    payload = blueprint.payload
    assert payload.schema_version == "1.0.0"
    assert payload.rubric_id == specification.rubric_id
    assert payload.rubric_version == specification.rubric_version
    assert payload.rubric_fingerprint == sha256_fingerprint(specification)
    assert payload.criterion_id == "business_materiality"
    assert payload.construct == "decision_consequence"
    assert payload.score_levels == specification.criteria[0].levels
    assert payload.task_family_id == "evidence_issue_analysis"
    assert payload.required_reasoning == (
        "identify_support",
        "identify_counterevidence",
    )
    assert payload.evidence_mode_id == "bounded_source_packet"
    assert payload.minimum_sources == 1
    assert payload.maximum_sources == 3
    assert payload.allowed_source_types == (
        "enterprise_report",
        "customer_record",
    )
    assert payload.difficulty_band_id == "direct_evidence_band"
    assert payload.locale == "ko-KR"
    assert payload.response_format == "structured_json_record"
    assert payload.prohibited_patterns == (
        "emotion_only_rationale",
        "invented_evidence",
    )
    assert payload.generation_rules == ("preserve_counterevidence",)
    assert blueprint.blueprint_fingerprint == sha256_fingerprint(payload)
    assert blueprint.blueprint_id == (
        "item_blueprint_" + blueprint.blueprint_fingerprint.removeprefix("sha256:")
    )


def test_blueprint_fingerprint_changes_with_nested_axis_constraints():
    """Nested evidence and difficulty changes cannot reuse an old item identity."""
    original = _valid_specification()
    changed_evidence = replace(
        original,
        evidence_modes=(replace(original.evidence_modes[0], maximum_sources=4),),
    )
    changed_difficulty = replace(
        original,
        difficulty_bands=(
            replace(original.difficulty_bands[0], constraints=("multi_issue",)),
        ),
    )
    original_id = compile_item_blueprints(original, max_blueprints=1)[0].blueprint_id
    assert compile_item_blueprints(changed_evidence, max_blueprints=1)[0].blueprint_id != original_id
    assert compile_item_blueprints(changed_difficulty, max_blueprints=1)[0].blueprint_id != original_id


def test_compiled_envelope_validation_rejects_invalid_identity_syntax():
    """Content-addressed envelope fields validate before contract construction."""
    payload = compile_item_blueprints(_valid_specification(), max_blueprints=1)[0].payload
    with pytest.raises(ValueError, match="blueprint_id.*snake_case"):
        CompiledBlueprint("invalid", "sha256:" + "0" * 64, payload)
    with pytest.raises(ValueError, match="blueprint_fingerprint"):
        CompiledBlueprint("item_blueprint_" + "0" * 64, "bad", payload)


def test_generated_item_schema_is_strict_and_returns_an_independent_copy():
    """Provider adapters receive a strict Draft 2020-12 schema without shared mutation."""
    schema = generated_item_output_schema()
    assert schema["$schema"] == GENERATED_ITEM_SCHEMA_DRAFT
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    required = set(schema["required"])
    assert required == {
        "item_id",
        "item_text",
        "response_format",
        "evidence_references",
        "rubric_alignment",
        "difficulty_rationale",
        "scoring_guidance",
        "prohibited_pattern_checks",
    }
    assert schema["properties"]["rubric_alignment"]["items"]["additionalProperties"] is False
    assert schema["properties"]["prohibited_pattern_checks"]["items"]["additionalProperties"] is False
    schema["properties"]["item_text"]["minLength"] = 99
    assert generated_item_output_schema()["properties"]["item_text"]["minLength"] == 1


def test_generation_contract_is_deterministic_provider_neutral_and_traceable():
    """A blueprint compiles to one immutable provider-neutral contract."""
    blueprint = compile_item_blueprints(_valid_specification(), max_blueprints=1)[0]
    contract = build_generation_contract(blueprint)
    repeated = build_generation_contract(blueprint)
    assert contract == repeated
    assert isinstance(contract, GenerationContract)
    assert isinstance(contract.payload, GenerationContractPayload)
    assert contract.contract_id.startswith("generation_contract_")
    assert contract.contract_fingerprint == sha256_fingerprint(contract.payload)
    assert contract.payload.blueprint_id == blueprint.blueprint_id
    assert contract.payload.blueprint_fingerprint == blueprint.blueprint_fingerprint
    assert contract.payload.blueprint == blueprint.payload
    assert contract.payload.generation_instructions == (
        "use_only_supplied_blueprint_evidence",
        "emit_exactly_one_generated_item_record",
        "do_not_add_unspecified_properties",
        "preserve_rubric_and_evidence_traceability",
        "report_prohibited_pattern_checks",
    )
    output_schema = json.loads(contract.payload.structured_output_schema_json)
    assert output_schema == generated_item_output_schema()
    contract_text = canonical_json(contract)
    for provider_term in ("openai", "anthropic", "gemini", "temperature", "api_key"):
        assert provider_term not in contract_text.lower()
    with pytest.raises(FrozenInstanceError):
        contract.contract_id = "changed_contract"  # type: ignore[misc]


def test_generation_contract_rejects_tampered_blueprint_fingerprint():
    """The builder recomputes content identity instead of trusting an envelope."""
    blueprint = compile_item_blueprints(_valid_specification(), max_blueprints=1)[0]
    tampered = replace(blueprint, blueprint_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="fingerprint.*does not match"):
        build_generation_contract(tampered)


def test_generation_contract_rejects_tampered_blueprint_identifier():
    """A syntactically valid but incorrect blueprint id is also detected."""
    blueprint = compile_item_blueprints(_valid_specification(), max_blueprints=1)[0]
    tampered = replace(blueprint, blueprint_id="item_blueprint_" + "0" * 64)
    with pytest.raises(ValueError, match="blueprint_id.*does not match"):
        build_generation_contract(tampered)


def test_generation_contract_requires_a_compiled_blueprint():
    """The contract builder rejects unvalidated arbitrary objects."""
    with pytest.raises(ValueError, match="CompiledBlueprint"):
        build_generation_contract({"blueprint_id": "bad_input"})


def test_generation_envelope_validation_rejects_invalid_identity_syntax():
    """Generation envelopes apply the same content-addressing syntax guards."""
    blueprint = compile_item_blueprints(_valid_specification(), max_blueprints=1)[0]
    payload = GenerationContractPayload(
        schema_version="1.0.0",
        blueprint_id=blueprint.blueprint_id,
        blueprint_fingerprint=blueprint.blueprint_fingerprint,
        blueprint=blueprint.payload,
        generation_instructions=("use_only_supplied_blueprint_evidence",),
        structured_output_schema_json=canonical_json(generated_item_output_schema()),
    )
    with pytest.raises(ValueError, match="contract_id.*snake_case"):
        GenerationContract("invalid", "sha256:" + "0" * 64, payload)
    with pytest.raises(ValueError, match="contract_fingerprint"):
        GenerationContract("generation_contract_" + "0" * 64, "bad", payload)


def test_payload_types_are_public_and_canonicalizable():
    """MSA consumers can persist public payload types without private adapters."""
    blueprint = compile_item_blueprints(_valid_specification(), max_blueprints=1)[0]
    assert isinstance(blueprint.payload, BlueprintPayload)
    decoded = json.loads(canonical_json(blueprint.payload))
    assert decoded["criterion_id"] == "business_materiality"
