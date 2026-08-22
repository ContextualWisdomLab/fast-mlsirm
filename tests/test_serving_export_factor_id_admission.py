"""Serving-export regressions for factor-to-dimension identity admission."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.types import FitResult, MLSIRMParams


class _HostileInteger:
    """Integer-like object whose coercion must not run during factor admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __int__(self) -> int:
        self.calls += 1
        raise AssertionError("caller-controlled integer conversion executed")

    def __index__(self) -> int:
        self.calls += 1
        raise AssertionError("caller-controlled index conversion executed")


def _result() -> FitResult:
    """Return a minimal converged two-item result for export-boundary tests."""
    params = MLSIRMParams(
        theta=np.zeros((1, 2), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model="MLS2PLM",
        optimizer="em",
        backend="rust",
        rust_device="cpu",
        objective=0.0,
        loglik_trace=[0.0],
        objective_trace=[],
        convergence_status="converged",
        n_iter=1,
    )


def _core_must_not_be_discovered() -> None:
    raise AssertionError("invalid serving evidence reached compiled-core discovery")


@pytest.mark.parametrize(
    "factor_id",
    [
        np.array([0, np.uint64(2**63)], dtype=np.uint64),
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([0.0 + 0.0j, 1.0 + 0.5j], dtype=np.complex128),
        np.array([-1, 0], dtype=np.int64),
    ],
)
def test_export_rejects_lossy_factor_identity_before_native_discovery(
    monkeypatch, factor_id
):
    """Factor identities must not wrap, truncate, project, or go negative."""
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="factor_id"):
        serving.export_serving_bundle(_result(), ["q0", "q1"], factor_id)


def test_export_rejects_object_factor_identity_without_integer_callback(monkeypatch):
    """Object-backed factor evidence must fail before per-cell conversion."""
    hostile = _HostileInteger()
    factor_id = np.array([0, hostile], dtype=object)
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="factor_id"):
        serving.export_serving_bundle(_result(), ["q0", "q1"], factor_id)

    assert hostile.calls == 0


def test_export_preserves_trusted_integer_valued_factor_sequence(monkeypatch):
    """Ordinary trusted integer-valued factor evidence keeps its Rust payload."""
    seen: dict[str, object] = {}

    def _eapsum_tables(alpha, b, zeta, tau, factor_id, *args, **kwargs):
        seen["factor_id"] = factor_id
        return []

    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: SimpleNamespace(eapsum_tables=_eapsum_tables),
    )

    bundle = serving.export_serving_bundle(
        _result(),
        ["q0", "q1"],
        (np.uint8(0), 1.0),
    )

    factor_id = seen["factor_id"]
    assert isinstance(factor_id, np.ndarray)
    assert factor_id.dtype == np.int64
    assert factor_id.tolist() == [0, 1]
    assert bundle["n_dims"] == 2
    assert [item["factor_id"] for item in bundle["items"]] == [0, 1]


@pytest.mark.parametrize("control_name", ["q_theta", "q_xi"])
def test_export_rejects_hostile_quadrature_integer_before_native_discovery(
    monkeypatch, control_name
):
    """Quadrature integer protocols must not execute before package admission."""
    hostile = _HostileInteger()
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match=control_name):
        serving.export_serving_bundle(
            _result(),
            ["q0", "q1"],
            (0, 1),
            **{control_name: hostile},
        )

    assert hostile.calls == 0


def test_export_normalizes_trusted_numpy_quadrature_controls_for_json(
    monkeypatch, tmp_path
):
    """Trusted NumPy integer controls must produce a self-serializable bundle."""
    seen: dict[str, object] = {}

    def _eapsum_tables(alpha, b, zeta, tau, factor_id, *args, **kwargs):
        seen["q_theta"] = kwargs["q_theta"]
        seen["q_xi"] = kwargs["q_xi"]
        return []

    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: SimpleNamespace(eapsum_tables=_eapsum_tables),
    )
    path = tmp_path / "bundle.json"

    bundle = serving.export_serving_bundle(
        _result(),
        ["q0", "q1"],
        (0, 1),
        path=path,
        q_theta=np.int64(21),
        q_xi=np.uint8(11),
    )

    assert type(bundle["quadrature"]["q_theta"]) is int
    assert type(bundle["quadrature"]["q_xi"]) is int
    assert type(seen["q_theta"]) is int
    assert type(seen["q_xi"]) is int
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["quadrature"] == {"q_theta": 21, "q_xi": 11}
