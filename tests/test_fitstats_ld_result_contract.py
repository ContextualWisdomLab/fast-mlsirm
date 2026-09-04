"""Public result-envelope contracts for local-dependence diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats_module


def _fixture():
    """Return a minimal admitted two-item LD request."""
    responses = np.zeros((20, 2), dtype=np.float64)
    factor_id = np.zeros(2, dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(2),
        b=np.zeros(2),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    return responses, factor_id, params


def _call(monkeypatch, payload):
    """Run the public boundary against one exact fake-native payload."""

    class Core:
        def ld_indices(self, *_args, **_kwargs):
            return payload

    monkeypatch.setattr(fitstats_module, "_core_module", lambda: Core())
    responses, factor_id, params = _fixture()
    return fitstats_module.ld_indices(
        responses,
        factor_id,
        params,
        "MIRT",
        q_theta=7,
        q_xi=3,
    )


def test_ld_indices_replays_parallel_native_vector_shape(monkeypatch):
    """X2/G2 evidence must describe the same ordered item-pair surface."""
    with pytest.raises(RuntimeError, match="matching one-dimensional pair vectors"):
        _call(
            monkeypatch,
            {"x2_signed": [0.0, 1.0], "g2_signed": [0.0]},
        )


def test_ld_indices_rejects_nontriangular_native_pair_count(monkeypatch):
    """A pair vector must have a cardinality representable as n_items choose two."""
    with pytest.raises(RuntimeError, match="triangular pair count"):
        _call(
            monkeypatch,
            {"x2_signed": [0.0, 1.0], "g2_signed": [0.0, 1.0]},
        )


def test_ld_indices_rejects_infinite_native_statistics(monkeypatch):
    """Undefined pairs may be NaN, but a live Rust statistic cannot be infinite."""
    with pytest.raises(RuntimeError, match="finite or NaN"):
        _call(
            monkeypatch,
            {"x2_signed": [np.inf], "g2_signed": [0.0]},
        )


def test_ld_indices_accepts_nan_for_undefined_native_pair(monkeypatch):
    """The fewer-than-20 joint-observation state remains explicit NaN evidence."""
    result = _call(
        monkeypatch,
        {"x2_signed": [np.nan], "g2_signed": [np.nan]},
    )
    assert result["x2_signed"].shape == (1,)
    assert result["g2_signed"].shape == (1,)
    assert np.isnan(result["x2_signed"][0])
    assert np.isnan(result["g2_signed"][0])
