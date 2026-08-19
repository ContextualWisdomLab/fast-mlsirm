"""Callback-safety regressions for IRT experiment-readiness controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.irt_contract import validate_irt_experiment_readiness


class _HostileInt(int):
    calls = 0

    def _trip(self):
        type(self).calls += 1
        raise AssertionError("caller integer callback executed")

    def __int__(self):
        self._trip()

    def __index__(self):
        self._trip()

    def __lt__(self, other):
        self._trip()

    def __le__(self, other):
        self._trip()

    def __gt__(self, other):
        self._trip()

    def __ge__(self, other):
        self._trip()


def _ready_matrix() -> list[list[int]]:
    """Return a small dichotomous matrix that satisfies readiness defaults."""
    return [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]


@pytest.mark.parametrize(
    "control_name, extra_kwargs",
    [
        ("min_persons", {}),
        ("min_observed_per_item", {}),
        ("min_item_distinct_values", {}),
        ("min_items_per_factor", {"factor_ids": ("g", "f")}),
    ],
)
def test_readiness_rejects_integer_subclasses_without_callbacks(
    control_name: str,
    extra_kwargs: dict[str, object],
) -> None:
    _HostileInt.calls = 0
    kwargs = dict(extra_kwargs)
    kwargs[control_name] = _HostileInt(1 if control_name == "min_items_per_factor" else 5)

    with pytest.raises(TypeError, match=control_name):
        validate_irt_experiment_readiness(
            _ready_matrix(),
            "dichotomous",
            **kwargs,
        )

    assert _HostileInt.calls == 0


def test_readiness_preserves_concrete_numpy_integer_controls() -> None:
    matrix = validate_irt_experiment_readiness(
        _ready_matrix(),
        "dichotomous",
        min_persons=np.int32(5),
        min_observed_per_item=np.uint16(3),
        min_item_distinct_values=np.int64(2),
        factor_ids=("g", "f"),
        min_items_per_factor=np.uint8(1),
    )

    assert matrix.shape == (5, 2)
