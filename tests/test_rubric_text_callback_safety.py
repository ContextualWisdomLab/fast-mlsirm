"""Trust-boundary regressions for public rubric text fields."""

from __future__ import annotations

import pytest

from fast_mlsirm.rubric import (
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
)


class _HostileString(str):
    """Caller string subclass whose text callback must never execute."""

    calls: list[str] = []

    def strip(self, chars: str | None = None) -> str:
        type(self).calls.append("strip")
        return super().strip(chars)


def _levels() -> tuple[RubricLevel, RubricLevel]:
    """Return one minimal valid two-level rubric scale."""
    return (
        RubricLevel(
            score=0,
            label="not demonstrated",
            descriptor="The evidence does not demonstrate the construct.",
            observable_indicators=("No supported evidence is present.",),
        ),
        RubricLevel(
            score=1,
            label="demonstrated",
            descriptor="The evidence demonstrates the construct.",
            observable_indicators=("Supported evidence is present.",),
        ),
    )


def test_rubric_level_rejects_string_subclass_without_strip_callback() -> None:
    """Public score-level text admission must reject caller string subclasses."""
    _HostileString.calls = []

    with pytest.raises(ValueError, match="label must be a string"):
        RubricLevel(
            score=0,
            label=_HostileString("hostile label"),
            descriptor="A valid descriptor.",
            observable_indicators=("A valid observable indicator.",),
        )

    assert _HostileString.calls == []


def test_rubric_specification_rejects_identifier_subclass_without_callback() -> None:
    """Shared text normalization must also protect identifier-backed fields."""
    _HostileString.calls = []

    with pytest.raises(ValueError, match="rubric_id must be a string"):
        RubricSpecification(
            rubric_id=_HostileString("hostile_rubric"),
            construct_id="target_construct",
            construct_definition="A bounded construct definition.",
            response_format=ResponseFormat.ORDINAL_RATING,
            levels=_levels(),
            task_families=("target_task",),
            evidence_requirements=("Observe the target behavior.",),
        )

    assert _HostileString.calls == []


def test_rubric_text_fields_preserve_exact_builtin_string_normalization() -> None:
    """Exact built-in strings retain the existing trim-and-normalize behavior."""
    level = RubricLevel(
        score=0,
        label="  demonstrated  ",
        descriptor="  Observable evidence.  ",
        observable_indicators=("  Supported behavior.  ",),
    )

    assert level.label == "demonstrated"
    assert level.descriptor == "Observable evidence."
    assert level.observable_indicators == ("Supported behavior.",)
