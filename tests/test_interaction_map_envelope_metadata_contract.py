"""Fail-closed public metadata contract for the Rust interaction-map envelope."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version as distribution_version

import numpy as np
import pytest

import fast_mlsirm.interaction_map_envelope as envelope_module


class _FakeCore:
    """Return only Rust-owned metadata so payload reads prove their ordering."""

    def __init__(self, result: Mapping[str, object]) -> None:
        self._result = dict(result)

    def residual_interaction_map_envelope(self, *_args: object) -> dict[str, object]:
        return dict(self._result)


def _current_metadata() -> dict[str, object]:
    return {
        "schema_version": "fast-mlsirm.residual-interaction-map.v1",
        "algorithm_id": "gabriel-complete-case-symmetric-residual-map.v1",
        "implementation_version": distribution_version("fast-mlsirm"),
        "calculation_provenance": "mlsirm-core::interaction_map::residual_interaction_map",
        "requested_axis_count": 1,
        "cell_extrema_tie_policy": "lexicographic-first-original-index",
        "finite_value_status": True,
    }


@pytest.mark.parametrize(
    ("field", "foreign_value", "message"),
    [
        ("schema_version", "fast-mlsirm.residual-interaction-map.v2", "schema version"),
        ("algorithm_id", "foreign-algorithm.v1", "algorithm"),
        ("implementation_version", "999.0.0", "implementation version"),
        ("calculation_provenance", "foreign::implementation", "calculation provenance"),
        ("requested_axis_count", 2, "requested axis count"),
        ("cell_extrema_tie_policy", "last-cell-wins", "tie policy"),
        ("finite_value_status", False, "finite-value status"),
    ],
)
def test_public_binding_rejects_foreign_rust_metadata_before_payload_marshalling(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    foreign_value: object,
    message: str,
) -> None:
    """A stale/foreign extension must fail before any numerical payload key is read."""
    metadata = _current_metadata()
    metadata[field] = foreign_value
    monkeypatch.setattr(envelope_module, "interaction_map_core", lambda: _FakeCore(metadata))

    with pytest.raises(RuntimeError, match=message):
        envelope_module.residual_interaction_map_envelope(
            np.ones((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            person_ids=["person-a"],
            item_ids=["item-a"],
            axis_count=1,
        )


def test_public_binding_rejects_foreign_input_digest_before_payload_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current-looking Rust result must be bound to the exact validated request."""
    metadata = _current_metadata()
    metadata["input_digest"] = "0" * 64
    monkeypatch.setattr(envelope_module, "interaction_map_core", lambda: _FakeCore(metadata))

    with pytest.raises(RuntimeError, match="input digest"):
        envelope_module.residual_interaction_map_envelope(
            np.ones((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            person_ids=["person-a"],
            item_ids=["item-a"],
            axis_count=1,
        )