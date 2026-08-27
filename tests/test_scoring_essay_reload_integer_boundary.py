"""Regression tests for essay integer safety across module reloads."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring import essay as essay_package
from fast_mlsirm.scoring.essay import contracts as essay_contracts
from fast_mlsirm.scoring.essay._integer_safety import install as install_integer_safety


class _HostileIndex:
    """Caller integer protocol that records executable conversion."""

    callbacks: list[str] = []

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 5_000


def _prompt(maximum_response_characters: object):
    """Build one valid prompt with a targeted integer-control override."""
    return essay_package.build_essay_prompt(
        prompt_id="argument_prompt",
        task_family_id="essay_review",
        prompt_content_fingerprint="1" * 64,
        language_id="english_language",
        genre_id="argument_genre",
        maximum_response_characters=maximum_response_characters,
        maximum_response_units=1_000,
    )


def test_contracts_reload_does_not_restore_integer_protocol_execution() -> None:
    """Reloading the implementation module must not weaken public integer admission."""
    _HostileIndex.callbacks.clear()
    importlib.reload(essay_contracts)
    try:
        with pytest.raises(AssessmentSpecError) as caught:
            _prompt(_HostileIndex())
        assert caught.value.code == "invalid_maximum_response_characters"
        assert _HostileIndex.callbacks == []
    finally:
        install_integer_safety(essay_contracts)


def test_contracts_reload_preserves_exact_numpy_integer_compatibility() -> None:
    """Reloaded source validation still accepts supported concrete NumPy integers."""
    importlib.reload(essay_contracts)
    try:
        prompt = _prompt(np.uint32(5_000))
        assert type(prompt.maximum_response_characters) is int
        assert prompt.maximum_response_characters == 5_000
    finally:
        install_integer_safety(essay_contracts)
