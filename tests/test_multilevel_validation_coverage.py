"""Edge contracts for multilevel validation and sealed aggregate replay.

These cases exercise failure paths that are intentionally difficult to reach
through the happy-path builders: malformed iterators, invalid UTF-8, direct
dataclass construction, and a replay that produces a different sealed child.
"""

from __future__ import annotations

import pytest

import fast_mlsirm.multilevel._validation as validation
import fast_mlsirm.multilevel.contracts as contracts
from fast_mlsirm.multilevel import (
    ContextMembership,
    LongitudinalStateSpec,
    TemporalOccasion,
    build_context_membership,
    build_context_membership_design,
    build_longitudinal_design,
    build_longitudinal_state_spec,
    build_temporal_occasion,
)


def _error_code(callback) -> str:
    """Return a structured contract error code from one callback."""
    with pytest.raises(validation.MultilevelContractError) as caught:
        callback()
    return caught.value.code


def test_validation_rejects_bad_schema_identifier_and_fingerprint() -> None:
    """Scalar contract fields reject invalid values without coercion."""
    assert _error_code(lambda: validation.schema_version("0.0")) == "invalid_schema_version"
    assert _error_code(lambda: validation.descriptive_identifier("one", "item_id")) == "invalid_item_id"
    assert _error_code(lambda: validation.fingerprint("ABC", "artifact_fingerprint")) == "invalid_artifact_fingerprint"


def test_validation_rejects_invalid_utf8_without_reflecting_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lone surrogate is rejected through the explicit UTF-8 guard."""

    class AcceptingPattern:
        """Permit the synthetic surrogate through the lexical test."""

        def fullmatch(self, _value: str) -> object:
            """Return a successful match sentinel."""
            return object()

    monkeypatch.setattr(validation, "_IDENTIFIER_PATTERN", AcceptingPattern())
    assert _error_code(lambda: validation.descriptive_identifier("\ud800", "item_id")) == "invalid_item_id"


def test_bounded_values_redacts_text_iterators_and_short_collections() -> None:
    """Collections reject text, iteration failures, and insufficient values."""
    for value in ("text", b"text", bytearray(b"text")):
        assert _error_code(lambda value=value: validation.bounded_values(value, "items", minimum=1, maximum=2)) == "invalid_items"

    class BadIterable:
        """Raise while asking for an iterator."""

        def __iter__(self):
            """Raise an ordinary exception that must be redacted."""
            raise ValueError("private iterator value")

    class BadIterator:
        """Raise while materializing the first element."""

        def __iter__(self):
            """Return this object as its iterator."""
            return self

        def __next__(self):
            """Raise an ordinary exception that must be redacted."""
            raise ValueError("private next value")

    for value in (BadIterable(), BadIterator(), []):
        assert _error_code(lambda value=value: validation.bounded_values(value, "items", minimum=1, maximum=2)) == "invalid_items"


def test_canonical_json_rejects_non_serializable_content() -> None:
    """Artifact hashing does not invoke arbitrary fallback stringification."""
    assert _error_code(lambda: validation.canonical_json(object())) == "invalid_canonical_content"


def test_direct_leaf_construction_is_not_an_authorized_factory_path() -> None:
    """All sealed leaf dataclasses require their private construction token."""
    assert _error_code(
        lambda: ContextMembership(
            "observation_one", "team_context", "team_one", 1.0, "a" * 64
        )
    ) == "unverified_context_membership"
    assert _error_code(
        lambda: TemporalOccasion(
            "respondent_one", "occasion_one", 0, 0, "b" * 64
        )
    ) == "unverified_temporal_occasion"
    assert _error_code(
        lambda: LongitudinalStateSpec(
            "random_intercept_slope", None, False
        )
    ) == "unverified_longitudinal_state_spec"


def test_membership_aggregate_rejects_a_non_exact_child() -> None:
    """An aggregate cannot be made valid by replacing a sealed child object."""
    edge = build_context_membership(
        observation_id="observation_one",
        context_dimension_id="team_context",
        context_id="team_one",
        membership_weight=1.0,
        membership_revision_fingerprint="c" * 64,
    )
    design = build_context_membership_design((edge,))
    object.__setattr__(design, "memberships", (object(),))
    assert _error_code(lambda: design.design_fingerprint) == "context_membership_design_integrity_mismatch"


def test_builders_reject_non_exact_collection_members() -> None:
    """Factory boundaries reject foreign children before replay or sorting."""
    assert _error_code(
        lambda: build_context_membership_design((object(),))
    ) == "invalid_context_membership"

    occasion = build_temporal_occasion(
        respondent_id="respondent_one",
        occasion_id="occasion_one",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="a" * 64,
    )
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    assert _error_code(
        lambda: build_longitudinal_design(occasions=(occasion,), state_spec=object())
    ) == "invalid_longitudinal_state_spec"
    assert _error_code(
        lambda: build_longitudinal_design(
            occasions=(occasion, object()), state_spec=state
        )
    ) == "invalid_temporal_occasion"


def test_longitudinal_aggregate_rejects_non_exact_state_or_occasion() -> None:
    """The longitudinal aggregate checks exact child classes before hashing."""
    occasion = build_temporal_occasion(
        respondent_id="respondent_one",
        occasion_id="occasion_one",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="d" * 64,
    )
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    design = build_longitudinal_design(occasions=(occasion,), state_spec=state)

    object.__setattr__(design, "state_spec", object())
    assert _error_code(lambda: design.design_fingerprint) == "longitudinal_design_integrity_mismatch"

    design = build_longitudinal_design(occasions=(occasion,), state_spec=state)
    object.__setattr__(design, "occasions", (object(),))
    assert _error_code(lambda: design.design_fingerprint) == "longitudinal_design_integrity_mismatch"


def test_replay_detects_different_sealed_children(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuilding a child with different content cannot preserve the old seal."""
    edge = build_context_membership(
        observation_id="observation_one",
        context_dimension_id="team_context",
        context_id="team_one",
        membership_weight=1.0,
        membership_revision_fingerprint="e" * 64,
    )
    other_edge = build_context_membership(
        observation_id="observation_two",
        context_dimension_id="team_context",
        context_id="team_one",
        membership_weight=1.0,
        membership_revision_fingerprint="f" * 64,
    )
    monkeypatch.setattr(contracts, "build_context_membership", lambda **_kwargs: other_edge)
    assert _error_code(lambda: build_context_membership_design((edge,))) == "context_membership_integrity_mismatch"

    occasion = build_temporal_occasion(
        respondent_id="respondent_one",
        occasion_id="occasion_one",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="1" * 64,
    )
    other_occasion = build_temporal_occasion(
        respondent_id="respondent_two",
        occasion_id="occasion_two",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="2" * 64,
    )
    monkeypatch.setattr(contracts, "build_temporal_occasion", lambda **_kwargs: other_occasion)
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    assert _error_code(lambda: build_longitudinal_design(occasions=(occasion,), state_spec=state)) == "temporal_occasion_integrity_mismatch"


def test_state_spec_replay_detects_different_sealed_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state specification replay has the same anti-substitution guarantee."""
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    other = build_longitudinal_state_spec(state_kind="stationary_autoregressive", autoregressive_coefficient=0.25)
    monkeypatch.setattr(contracts, "build_longitudinal_state_spec", lambda **_kwargs: other)
    occasion = build_temporal_occasion(
        respondent_id="respondent_one",
        occasion_id="occasion_one",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint="3" * 64,
    )
    assert _error_code(lambda: build_longitudinal_design(occasions=(occasion,), state_spec=state)) == "longitudinal_state_spec_integrity_mismatch"


def test_longitudinal_design_rejects_revision_fingerprint_reuse() -> None:
    """One revision fingerprint cannot identify two distinct occasions."""
    revision = "4" * 64
    first = build_temporal_occasion(
        respondent_id="respondent_one",
        occasion_id="occasion_one",
        sequence_index=0,
        time_offset_milliseconds=0,
        occasion_revision_fingerprint=revision,
    )
    second = build_temporal_occasion(
        respondent_id="respondent_one",
        occasion_id="occasion_two",
        sequence_index=1,
        time_offset_milliseconds=1,
        occasion_revision_fingerprint=revision,
    )
    state = build_longitudinal_state_spec(state_kind="random_intercept_slope")
    assert _error_code(
        lambda: build_longitudinal_design(occasions=(first, second), state_spec=state)
    ) == "occasion_revision_conflict"
