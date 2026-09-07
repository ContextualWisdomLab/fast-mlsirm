"""Binary64 contract for marginal EAP posterior-weight reductions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MARGINAL_SOURCE = (
    Path(__file__).parents[1] / "python" / "fast_mlsirm" / "estimators" / "marginal.py"
)


def test_eap_accumulation_keeps_established_multiply_then_reduce_order() -> None:
    """Reject ternary einsum reassociation that changes an ordinary finite EAP moment."""
    post = np.array(
        [[[[
            float.fromhex("0x1.7b8c142cc06a2p-17"),
            float.fromhex("0x1.090caa721bf80p-12"),
        ], [
            float.fromhex("0x1.fdb7bcaa52a4cp-1"),
            float.fromhex("0x1.12d31a2575828p-8"),
        ]]]],
        dtype=np.float64,
    )
    w_outer = np.array([float.fromhex("0x1.8e659e3c1bd07p+6")], dtype=np.float64)
    theta_s = np.array(
        [[[float.fromhex("0x1.7158af669e674p+1"), float.fromhex("-0x1.9756342da4036p+1")]]],
        dtype=np.float64,
    )

    wpost = post * w_outer[:, None, None, None]
    established = np.einsum("pdtx,pdt->pd", wpost, theta_s, optimize=True)
    reassociated = np.einsum("pdtx,p,pdt->pd", post, w_outer, theta_s, optimize=True)

    assert established[0, 0].hex() == "-0x1.3ccbff7f78810p+8"
    assert reassociated[0, 0].hex() == "-0x1.3ccbff7f7880fp+8"
    assert not np.array_equal(established, reassociated)

    source = MARGINAL_SOURCE.read_text(encoding="utf-8")
    assert "wpost = post * w_outer[:, None, None, None]" in source
    assert '"pdtx,p,pdt->pd"' not in source
