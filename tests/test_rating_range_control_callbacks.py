"""Fail-first contracts for paired rating-range control marshalling."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rating_range as rating_range


_AUTOMATED = np.array([0, 1], dtype=np.int64)
_REFERENCE = np.array([0, 1], dtype=np.int64)


def _reject_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Make native-core discovery observable and forbidden for invalid controls."""
    calls: list[bool] = []

    def discover_core() -> object:
        calls.append(True)
        raise AssertionError("native core discovered before control validation")

    monkeypatch.setattr(rating_range, "rating_range_core", discover_core)
    return calls


@pytest.mark.parametrize("kind", ["python", "numpy"])
def test_category_count_subclasses_fail_without_callbacks_or_core_discovery(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Caller-defined integer subclasses are data, not scalar-control authority."""
    discovery_calls = _reject_core_discovery(monkeypatch)
    callbacks: list[str] = []

    class HostilePythonInt(int):
        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller-controlled __int__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("caller-controlled __repr__ executed")

    class HostileNumpyInt(np.int64):
        def __int__(self) -> int:
            callbacks.append("int")
            raise AssertionError("caller-controlled NumPy __int__ executed")

        def __repr__(self) -> str:
            callbacks.append("repr")
            raise AssertionError("caller-controlled NumPy __repr__ executed")

    value = HostilePythonInt(2) if kind == "python" else HostileNumpyInt(2)

    with pytest.raises(
        ValueError,
        match=r"^category_count must be an integer between 2 and 1000$",
    ):
        rating_range.paired_rating_range_evidence(
            _AUTOMATED,
            _REFERENCE,
            category_count=value,
        )

    assert callbacks == []
    assert discovery_calls == []


@pytest.mark.parametrize(
    "scalar_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_exact_numpy_integer_category_counts_remain_accepted(
    monkeypatch: pytest.MonkeyPatch,
    scalar_type: type[np.integer],
) -> None:
    """Exact supported NumPy integer scalars normalize before Rust dispatch."""
    seen: list[int] = []

    class RecordingCore:
        def paired_rating_range_evidence(
            self,
            automated: np.ndarray,
            reference: np.ndarray,
            category_count: int,
        ) -> dict[str, object]:
            assert type(category_count) is int
            seen.append(category_count)
            return {
                "sample_size": 2,
                "automated_min": 0,
                "automated_max": 1,
                "reference_min": 0,
                "reference_max": 1,
                "automated_distinct_categories": 2,
                "reference_distinct_categories": 2,
                "automated_span": 1,
                "reference_span": 1,
                "automated_sd": 0.5,
                "reference_sd": 0.5,
                "span_ratio": 1.0,
                "distinct_category_ratio": 1.0,
                "sd_ratio": 1.0,
                "lower_endpoint_gap": 0,
                "upper_endpoint_gap": 0,
                "narrower_observed_support": False,
                "central_tendency_signal": False,
            }

    monkeypatch.setattr(rating_range, "rating_range_core", lambda: RecordingCore())

    result = rating_range.paired_rating_range_evidence(
        _AUTOMATED,
        _REFERENCE,
        category_count=scalar_type(2),
    )

    assert result.sample_size == 2
    assert seen == [2]
