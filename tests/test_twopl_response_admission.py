"""Trust-boundary regressions for compensatory 2PL response evidence."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.irt_contract import MAX_IRT_RESPONSE_CELLS
from fast_mlsirm.twopl import fit_2pl


class _ArrayProvider:
    """Top-level array provider that must never execute during admission."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.calls.append("array")
        raise AssertionError("response array callback executed")


class _FloatProvider:
    """Nested numeric provider that must never execute during admission."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __float__(self) -> float:
        self.calls.append("float")
        raise AssertionError("response float callback executed")


def _install_no_core(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install a core-discovery sentinel and return its call log."""

    core_calls: list[str] = []
    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: core_calls.append("core") or None,
    )
    return core_calls


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param(_ArrayProvider(), id="top-level-array-provider"),
        pytest.param([[_FloatProvider(), 1.0], [1.0, 0.0]], id="nested-float-provider"),
    ],
)
def test_fit_2pl_rejects_callback_bearing_response_evidence_before_native(
    monkeypatch: pytest.MonkeyPatch,
    responses: object,
) -> None:
    """Caller conversion hooks do not run while response evidence is admitted."""

    core_calls = _install_no_core(monkeypatch)

    with pytest.raises(ValueError, match="responses"):
        fit_2pl(responses)  # type: ignore[arg-type]

    if isinstance(responses, _ArrayProvider):
        assert responses.calls == []
    else:
        assert responses[0][0].calls == []
    assert core_calls == []


def test_fit_2pl_rejects_complex_response_before_real_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imaginary response evidence cannot be silently projected to float64."""

    core_calls = _install_no_core(monkeypatch)
    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        fit_2pl(responses)

    assert core_calls == []


def test_fit_2pl_rejects_extended_precision_value_that_would_round_to_binary_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wider real value must not become 1.0 merely because float64 narrows it."""

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no additional precision on this platform")

    core_calls = _install_no_core(monkeypatch)
    one = np.longdouble(1)
    widened = np.nextafter(one, np.longdouble(2), dtype=np.longdouble)
    responses = np.array([[0, widened], [1, 0]], dtype=np.longdouble)

    with pytest.raises(ValueError, match="integer category"):
        fit_2pl(responses)

    assert core_calls == []


def test_fit_2pl_rejects_oversized_logical_matrix_before_dense_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small-backed broadcast view is bounded by logical cells before scanning/copying."""

    core_calls = _install_no_core(monkeypatch)
    responses = np.broadcast_to(
        np.array([[0.0]], dtype=np.float64),
        (MAX_IRT_RESPONSE_CELLS + 1, 1),
    )

    with pytest.raises(ValueError, match="at most 20,000,000 cells"):
        fit_2pl(responses)

    assert core_calls == []


def test_fit_2pl_preserves_trusted_builtin_and_numpy_response_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe historical matrix inputs still reach Rust as package-owned float64 evidence."""

    captured: dict[str, np.ndarray] = {}

    class _Core:
        def fit_2pl(
            self,
            yy: np.ndarray,
            observed: np.ndarray,
            _pattern: np.ndarray,
            n_persons: int,
            n_items: int,
            n_dims: int,
            _q: int,
            _estimate_corr: bool,
            _max_iter: int,
            _tol: float,
            _node_rule: str,
            _xi_points: int,
            _xi_seed: int,
        ) -> dict[str, object]:
            captured["yy"] = yy.copy()
            captured["observed"] = observed.copy()
            return {
                "loading": np.ones(n_items * n_dims),
                "intercept": np.zeros(n_items),
                "theta": np.zeros(n_persons * n_dims),
                "corr": np.eye(n_dims),
                "loglik_trace": np.array([-2.0, -1.0]),
                "n_iter": 2,
                "converged": True,
                "n_parameters": n_items * (n_dims + 1),
                "termination_reason": "converged",
                "final_loglik_change": 1.0,
            }

    import fast_mlsirm.fitstats as fitstats

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    fit_2pl(
        [
            [np.bool_(False), np.int8(1)],
            [np.float32(1.0), float("nan")],
        ]
    )

    assert captured["yy"].dtype == np.float64
    assert captured["yy"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert captured["observed"].dtype == np.bool_
    assert captured["observed"].tolist() == [True, True, True, False]
