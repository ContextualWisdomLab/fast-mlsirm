"""Fail-first contracts for bounded polytomous integration-rule validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.gpcm import fit_gpcm
from fast_mlsirm.grm import fit_grm
from fast_mlsirm.nominal import fit_nominal


class _HostileNodeRule:
    """Invalid rule whose representation hooks must never run during validation."""

    def __str__(self) -> str:
        raise AssertionError("NODE_RULE_STR_SENTINEL")

    def __repr__(self) -> str:
        raise AssertionError("NODE_RULE_REPR_SENTINEL")


def _responses() -> np.ndarray:
    """Return a tiny valid three-category response matrix for public-fit preflight."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("fitter", [fit_grm, fit_gpcm, fit_nominal])
def test_public_polytomous_fit_rejects_hostile_node_rule_without_stringification(
    fitter,
) -> None:
    """Invalid integration controls fail with a package error before caller hooks run."""
    with pytest.raises(ValueError, match="node_rule must be one of the supported integration rules"):
        fitter(
            _responses(),
            3,
            model=1,
            q=7,
            max_iter=1,
            node_rule=_HostileNodeRule(),
        )
