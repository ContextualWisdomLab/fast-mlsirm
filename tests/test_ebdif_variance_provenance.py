"""Cross-field provenance checks for EBDIF native prior-variance results."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ebdif as ebdif
import fast_mlsirm.fitstats as fitstats


def _result(*, tau2: float, tau2_raw: float) -> dict[str, object]:
    """Return an otherwise-valid two-item native EBDIF payload."""

    return {
        "mu": 0.0,
        "tau2": tau2,
        "tau2_raw": tau2_raw,
        "weight": np.zeros(2, dtype=np.float64),
        "post_mean": np.zeros(2, dtype=np.float64),
        "post_var": np.zeros(2, dtype=np.float64),
        "cat_probs": np.array([0.0, 0.0, 1.0, 0.0, 0.0] * 2, dtype=np.float64),
    }


def _core_returning(result: dict[str, object]) -> object:
    """Return a fake native module exposing one deterministic payload."""

    class Core:
        @staticmethod
        def py_eb_mh_dif(mh: np.ndarray, se: np.ndarray) -> dict[str, object]:
            del mh, se
            return result

    return Core()


@pytest.mark.parametrize(
    ("tau2", "tau2_raw"),
    [
        (0.5, -0.1),
        (0.5, 0.25),
    ],
)
def test_native_floored_variance_must_match_pre_floor_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tau2: float,
    tau2_raw: float,
) -> None:
    """A stale core cannot publish a floored variance inconsistent with tau2_raw."""

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: _core_returning(_result(tau2=tau2, tau2_raw=tau2_raw)),
    )

    with pytest.raises(RuntimeError, match="invalid EBDIF Rust result payload"):
        ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])


def test_native_negative_zero_pre_floor_remains_zero_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The established negative-zero representation remains the zero-variance state."""

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: _core_returning(_result(tau2=0.0, tau2_raw=-0.0)),
    )

    result = ebdif.eb_mh_dif([0.1, -0.2], [0.3, 0.4])

    assert result.tau2 == 0.0
    assert result.tau2_raw == 0.0
