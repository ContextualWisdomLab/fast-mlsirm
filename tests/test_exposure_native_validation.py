"""Native validation coverage retained when Python mirrors Rust domains."""

from __future__ import annotations

import numpy as np
import pytest


def test_sprt_native_rejects_incoherent_error_rates() -> None:
    """Exercise the Rust alpha+beta guard without the public prevalidation layer."""

    from fast_mlsirm import _core

    a = np.array([1.0, 1.0], dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    c = np.zeros(2, dtype=np.float64)
    responses = np.array([1, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"alpha \+ beta"):
        _core.py_sprt_classify(
            a,
            b,
            c,
            responses,
            0.0,
            0.5,
            0.6,
            0.5,
        )
