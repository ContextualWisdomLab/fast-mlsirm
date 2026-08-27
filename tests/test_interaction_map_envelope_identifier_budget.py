"""Opaque interaction-map identifier resource-boundary regressions."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fast_mlsirm import residual_interaction_map_envelope


envelope_module = importlib.import_module("fast_mlsirm.interaction_map_envelope")


def test_oversized_person_ids_fail_before_copy_or_matrix_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identifier count is rejected from inert length metadata before matrix work."""
    monkeypatch.setattr(
        envelope_module,
        "_MAX_INTERACTION_MAP_IDENTIFIER_COUNT",
        2,
        raising=False,
    )

    def fail_matrix(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("matrix admission must not run for oversized identifiers")

    monkeypatch.setattr(envelope_module, "_trusted_matrix", fail_matrix)

    with pytest.raises(ValueError, match="person_ids identifier count exceeds 2"):
        residual_interaction_map_envelope(
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            person_ids=["person-a", "person-b", "person-c"],
            item_ids=["item-a"],
            axis_count=1,
        )


def test_identifier_character_budget_fails_before_matrix_or_digest_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate opaque-ID text is bounded before matrices or digest encoding."""
    monkeypatch.setattr(
        envelope_module,
        "_MAX_INTERACTION_MAP_IDENTIFIER_CHARACTERS",
        8,
        raising=False,
    )

    def fail_matrix(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("matrix admission must not run for oversized identifier text")

    monkeypatch.setattr(envelope_module, "_trusted_matrix", fail_matrix)

    with pytest.raises(
        ValueError,
        match="person_ids identifier character count exceeds 8",
    ):
        residual_interaction_map_envelope(
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            person_ids=["abcd", "efghi"],
            item_ids=["item-a"],
            axis_count=1,
        )


def test_single_identifier_length_is_bounded_before_matrix_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pathological opaque identifier cannot trigger unbounded UTF-8 work."""
    monkeypatch.setattr(
        envelope_module,
        "_MAX_INTERACTION_MAP_IDENTIFIER_LENGTH",
        8,
        raising=False,
    )

    def fail_matrix(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("matrix admission must not run for an oversized identifier")

    monkeypatch.setattr(envelope_module, "_trusted_matrix", fail_matrix)

    with pytest.raises(ValueError, match="person_ids identifier length exceeds 8"):
        residual_interaction_map_envelope(
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            person_ids=["123456789"],
            item_ids=["item-a"],
            axis_count=1,
        )


def test_identifier_budget_preserves_valid_small_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid identifier set at reduced boundaries still reaches Rust unchanged."""
    monkeypatch.setattr(
        envelope_module,
        "_MAX_INTERACTION_MAP_IDENTIFIER_COUNT",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        envelope_module,
        "_MAX_INTERACTION_MAP_IDENTIFIER_LENGTH",
        8,
        raising=False,
    )
    monkeypatch.setattr(
        envelope_module,
        "_MAX_INTERACTION_MAP_IDENTIFIER_CHARACTERS",
        16,
        raising=False,
    )

    result = residual_interaction_map_envelope(
        np.ones((2, 1), dtype=np.float64),
        np.ones((2, 1), dtype=np.float64),
        person_ids=["person-a", "person-b"],
        item_ids=["item-a"],
        axis_count=1,
    )

    assert result.retained_person_ids == ("person-a", "person-b")
    assert result.retained_item_ids == ("item-a",)
