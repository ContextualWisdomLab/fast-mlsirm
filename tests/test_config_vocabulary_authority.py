"""Regression coverage for immutable public FitConfig vocabulary authority."""

from __future__ import annotations

from collections.abc import MutableSet

import pytest

from fast_mlsirm.config import (
    FitConfig,
    VALID_ESTIMATORS,
    VALID_MODELS,
    VALID_OPTIMIZERS,
)


def _require_immutable_vocabulary(vocabulary: object, injected: str) -> None:
    """Prove callers cannot rewrite validation authority through exported constants."""
    try:
        with pytest.raises(AttributeError):
            vocabulary.add(injected)  # type: ignore[attr-defined]
    finally:
        if isinstance(vocabulary, MutableSet):
            vocabulary.discard(injected)


def test_fit_config_vocabularies_cannot_be_mutated_by_callers() -> None:
    """Public model, estimator, and optimizer vocabularies stay read-only."""
    _require_immutable_vocabulary(VALID_MODELS, "CALLER_INJECTED_MODEL")
    _require_immutable_vocabulary(VALID_ESTIMATORS, "caller_injected_estimator")
    _require_immutable_vocabulary(VALID_OPTIMIZERS, "caller_injected_optimizer")

    with pytest.raises(ValueError, match="model must be one of"):
        FitConfig(model="CALLER_INJECTED_MODEL")
    with pytest.raises(ValueError, match="estimator must be one of"):
        FitConfig(estimator="caller_injected_estimator")
    with pytest.raises(ValueError, match="optimizer must be one of"):
        FitConfig(optimizer="caller_injected_optimizer")
