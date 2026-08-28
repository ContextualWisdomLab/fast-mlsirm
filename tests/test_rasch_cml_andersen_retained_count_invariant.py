"""Result-integrity regressions for Andersen retained-person counts."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.rasch_cml import andersen_lr_test


def _binary() -> np.ndarray:
    """Return four admitted persons split across two Andersen groups."""

    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_andersen_rejects_retained_count_sum_above_admitted_persons(monkeypatch):
    """Disjoint group counts cannot claim more retained persons than were admitted."""

    class _Core:
        def andersen_lr_test(self, *args):
            del args
            return {
                "lr": 0.0,
                "df": 2,
                "p_value": 1.0,
                "n_used": [3, 3],
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    with pytest.raises(RuntimeError, match="invalid Andersen Rust result payload"):
        andersen_lr_test(_binary(), [0, 0, 1, 1])


def test_andersen_rejects_retained_count_above_group_capacity(monkeypatch):
    """A group cannot retain more informative persons than that group admitted."""

    class _Core:
        def andersen_lr_test(self, *args):
            del args
            return {
                "lr": 0.0,
                "df": 2,
                "p_value": 1.0,
                "n_used": [3, 1],
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    with pytest.raises(RuntimeError, match="invalid Andersen Rust result payload"):
        andersen_lr_test(_binary(), [0, 0, 1, 1])
