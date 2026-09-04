"""Cryptographic request-provenance contract for the interaction-map envelope."""

from __future__ import annotations

import numpy as np

from fast_mlsirm import residual_interaction_map_envelope


def _envelope(
    observed: object,
    expected: object,
    *,
    person_ids: list[str] | tuple[str, ...] = ("person-a", "person-b"),
    item_ids: list[str] | tuple[str, ...] = ("item-a", "item-b"),
    axis_count: int = 1,
):
    return residual_interaction_map_envelope(
        observed,
        expected,
        person_ids=person_ids,
        item_ids=item_ids,
        axis_count=axis_count,
    )


def test_input_digest_is_deterministic_across_equivalent_admitted_carriers() -> None:
    """Equivalent validated binary64 evidence must have one platform-stable SHA-256 identity."""
    observed_array = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    expected_array = np.ones((2, 2), dtype=np.float64)

    array_result = _envelope(observed_array, expected_array)
    sequence_result = _envelope(
        [[2.0, 0.0], [0.0, 2.0]],
        ((1.0, 1.0), (1.0, 1.0)),
        person_ids=["person-a", "person-b"],
        item_ids=["item-a", "item-b"],
    )

    assert len(array_result.input_digest) == 64
    assert set(array_result.input_digest) <= set("0123456789abcdef")
    assert sequence_result.input_digest == array_result.input_digest


def test_input_digest_changes_with_evidence_identity_or_axis_request() -> None:
    """Any validated request identity change must produce a different provenance digest."""
    observed = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    expected = np.ones((2, 2), dtype=np.float64)
    baseline = _envelope(observed, expected).input_digest

    changed_observed = observed.copy()
    changed_observed[0, 0] = 3.0
    assert _envelope(changed_observed, expected).input_digest != baseline
    assert (
        _envelope(
            observed,
            expected,
            person_ids=("person-a", "person-c"),
        ).input_digest
        != baseline
    )
    assert _envelope(observed, expected, axis_count=2).input_digest != baseline


def test_input_digest_canonicalizes_missing_nan_payload_and_sign_bits() -> None:
    """Equivalent missing-response NaNs must share one provenance identity."""
    nan_bits = (
        0x7FF8000000000001,
        0x7FF8000000000002,
        0xFFF8000000000001,
    )
    expected = np.ones((2, 2), dtype=np.float64)
    observed_variants = []
    for bits in nan_bits:
        missing = np.array([bits], dtype=np.uint64).view(np.float64)[0]
        observed_variants.append(
            np.array([[2.0, missing], [0.0, 2.0]], dtype=np.float64)
        )

    assert all(np.isnan(observed[0, 1]) for observed in observed_variants)
    assert {
        int(observed.view(np.uint64)[0, 1]) for observed in observed_variants
    } == set(nan_bits)
    assert len({_envelope(observed, expected).input_digest for observed in observed_variants}) == 1
