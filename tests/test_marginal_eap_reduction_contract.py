"""Binary64 contract for marginal EAP posterior-weight reductions."""

from __future__ import annotations

from pathlib import Path
import struct


MARGINAL_SOURCE = (
    Path(__file__).parents[1] / "python" / "fast_mlsirm" / "estimators" / "marginal.py"
)


def _round_binary64(value: float) -> float:
    """Force one explicit IEEE-754 binary64 rounding boundary."""
    return struct.unpack(">d", struct.pack(">d", value))[0]


def test_eap_accumulation_keeps_established_multiply_then_reduce_order() -> None:
    """Reject a reassociation that changes an ordinary finite EAP moment by one ULP."""
    post = (
        (
            float.fromhex("0x1.7b8c142cc06a2p-17"),
            float.fromhex("0x1.090caa721bf80p-12"),
        ),
        (
            float.fromhex("0x1.fdb7bcaa52a4cp-1"),
            float.fromhex("0x1.12d31a2575828p-8"),
        ),
    )
    weight = float.fromhex("0x1.8e659e3c1bd07p+6")
    theta = (
        float.fromhex("0x1.7158af669e674p+1"),
        float.fromhex("-0x1.9756342da4036p+1"),
    )

    established = 0.0
    reassociated_inner = 0.0
    for theta_index, row in enumerate(post):
        for probability in row:
            weighted_probability = _round_binary64(probability * weight)
            established_term = _round_binary64(
                weighted_probability * theta[theta_index]
            )
            established = _round_binary64(established + established_term)

            reassociated_term = _round_binary64(
                probability * theta[theta_index]
            )
            reassociated_inner = _round_binary64(
                reassociated_inner + reassociated_term
            )

    reassociated = _round_binary64(reassociated_inner * weight)

    assert established.hex() == "-0x1.3ccbff7f78810p+8"
    assert reassociated.hex() == "-0x1.3ccbff7f7880fp+8"
    assert established != reassociated

    source = MARGINAL_SOURCE.read_text(encoding="utf-8")
    assert "wpost = post * w_outer[:, None, None, None]" in source
    assert '"pdtx,p,pdt->pd"' not in source
