"""Regression coverage for Mokken native-result root snapshot provenance."""

from __future__ import annotations

import numpy as np

import fast_mlsirm.mokken as mokken


def _valid_native_payload() -> dict[str, object]:
    return {
        "hij": [float("nan"), 0.5, 0.5, float("nan")],
        "hi": [0.4, 0.4],
        "h": 0.4,
        "zij": [float("nan"), 1.0, 1.0, float("nan")],
        "zi": [1.0, 1.0],
        "z": 1.0,
    }


def test_native_coefficient_root_is_snapshotted_before_vector_validation(
    monkeypatch,
) -> None:
    result = _valid_native_payload()
    admitted_hi = result["hi"]
    replacement_hi = [0.9, 0.9]
    real_pairwise = mokken._native_pairwise_matrix
    pairwise_calls = 0

    def mutating_pairwise(value: object, n_items: int) -> np.ndarray:
        nonlocal pairwise_calls
        vector = real_pairwise(value, n_items)
        if pairwise_calls == 0:
            result["hi"] = replacement_hi
        pairwise_calls += 1
        return vector

    monkeypatch.setattr(mokken, "_native_pairwise_matrix", mutating_pairwise)

    coefficients = mokken._validated_native_coefficients(result, n_items=2)

    assert coefficients.hi.tolist() == admitted_hi
    assert coefficients.hi.tolist() != replacement_hi
