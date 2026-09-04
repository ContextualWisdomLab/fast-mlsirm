"""Numerical-route contract for the legacy Python marginal distance reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MARGINAL_SOURCE = (
    Path(__file__).parents[1] / "python" / "fast_mlsirm" / "estimators" / "marginal.py"
)


def test_rowwise_squared_distance_keeps_established_binary64_reduction() -> None:
    """An allocation optimization must not silently reassociate estimator arithmetic."""
    diff = np.array(
        [[
            float.fromhex("0x1.9a4c408bd8417p+0"),
            float.fromhex("0x1.7f38d14a8fe3bp-14"),
            float.fromhex("0x1.b02b980a9c79cp+2"),
        ]],
        dtype=np.float64,
    )

    established = np.sum(diff * diff, axis=1)
    proposed = np.einsum("ij,ij->i", diff, diff)

    assert established[0].hex() == "0x1.815656f4f071ap+5"
    assert proposed[0].hex() == "0x1.815656f4f0719p+5"
    assert not np.array_equal(established, proposed)

    source = MARGINAL_SOURCE.read_text(encoding="utf-8")
    assert "np.sum(diff * diff, axis=1)" in source
    assert "np.einsum('ij,ij->i', diff, diff)" not in source
