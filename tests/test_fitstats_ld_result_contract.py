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


def test_ld_indices_rejects_missing_native_pair_evidence(monkeypatch):
    """The public boundary must translate a missing required native field."""
    with pytest.raises(RuntimeError, match="missing required pair evidence"):
        _call(monkeypatch, {"x2_signed": [0.0]})


def test_ld_indices_rejects_extra_native_result_fields(monkeypatch):
    """An additive stale-core field cannot silently redefine the public envelope."""
    with pytest.raises(RuntimeError, match="contain exactly x2_signed and g2_signed"):
        _call(
            monkeypatch,
            {"x2_signed": [0.0], "g2_signed": [0.0], "legacy_flag": True},
        )


def test_ld_indices_replays_parallel_native_vector_shape(monkeypatch):
    """X2/G2 evidence must describe the same ordered item-pair surface."""
    with pytest.raises(RuntimeError, match="matching one-dimensional pair vectors"):
        _call(
            monkeypatch,
            {"x2_signed": [0.0, 1.0], "g2_signed": [0.0]},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"x2_signed": [0], "g2_signed": [0]},
        {"x2_signed": [[0.0]], "g2_signed": [[0.0]]},
    ],
)
def test_ld_indices_rejects_wrong_native_vector_representation(monkeypatch, payload):
    """Published pair vectors must be one-dimensional binary64 evidence."""
    with pytest.raises(RuntimeError, match="matching one-dimensional pair vectors"):
        _call(monkeypatch, payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"x2_signed": [], "g2_signed": []},
        {"x2_signed": [0.0, 1.0], "g2_signed": [0.0, 1.0]},
    ],
)
def test_ld_indices_rejects_invalid_native_pair_count(monkeypatch, payload):
    """A pair vector must have a positive n_items-choose-two cardinality."""
    with pytest.raises(RuntimeError, match="triangular pair count"):
        _call(monkeypatch, payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"x2_signed": [np.inf], "g2_signed": [0.0]},
        {"x2_signed": [0.0], "g2_signed": [-np.inf]},
    ],
)
def test_ld_indices_rejects_infinite_native_statistics(monkeypatch, payload):
    """Undefined pairs may be NaN, but a live Rust statistic cannot be infinite."""
    with pytest.raises(RuntimeError, match="finite or NaN"):
        _call(monkeypatch, payload)


def test_ld_indices_accepts_nan_and_owns_published_pair_vectors(monkeypatch):
    """Undefined NaNs survive while published arrays no longer alias native storage."""
    x2_native = np.array([np.nan], dtype=np.float64)
    g2_native = np.array([np.nan], dtype=np.float64)
    result = _call(
        monkeypatch,
        {"x2_signed": x2_native, "g2_signed": g2_native},
    )
    assert result["x2_signed"].shape == (1,)
    assert result["g2_signed"].shape == (1,)
    assert np.isnan(result["x2_signed"][0])
    assert np.isnan(result["g2_signed"][0])
    assert not np.shares_memory(result["x2_signed"], x2_native)
    assert not np.shares_memory(result["g2_signed"], g2_native)
