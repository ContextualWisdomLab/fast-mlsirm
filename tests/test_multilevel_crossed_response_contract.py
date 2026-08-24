"""Binary-response trust-boundary tests for crossed person-effect estimation."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import fast_mlsirm.multilevel.estimation as estimation
from fast_mlsirm.multilevel import (
    build_context_membership,
    build_context_membership_design,
)


def _revision(tag: str) -> str:
    """Return one deterministic content fingerprint for the test fixture."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _design():
    """Build the smallest identified one-classification membership design."""
    return build_context_membership_design(
        [
            build_context_membership(
                observation_id="person_alpha",
                context_dimension_id="school_membership",
                context_id="school_east",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("alpha-east"),
            ),
            build_context_membership(
                observation_id="person_beta",
                context_dimension_id="school_membership",
                context_id="school_west",
                membership_weight=1.0,
                membership_revision_fingerprint=_revision("beta-west"),
            ),
        ]
    )


@pytest.mark.parametrize("invalid_response", [0.5, 1.5, 2.0])
def test_nonbinary_observed_response_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: float,
) -> None:
    """Finite observed cells outside {0, 1} must never reach the Rust core."""
    core_discoveries = 0

    def _unexpected_core_discovery():
        nonlocal core_discoveries
        core_discoveries += 1
        raise AssertionError("native core must not be discovered for invalid responses")

    monkeypatch.setattr(estimation, "multilevel_core", _unexpected_core_discovery)
    responses = np.array([[invalid_response], [0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="binary responses"):
        estimation.estimate_crossed_person_effects(
            responses,
            _design(),
            item_intercepts=np.array([0.0], dtype=np.float64),
            device="cpu",
        )

    assert core_discoveries == 0
