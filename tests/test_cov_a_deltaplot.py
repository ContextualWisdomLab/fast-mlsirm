"""Coverage for Angoff delta-plot DIF detection (deltaplot.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.deltaplot import DeltaPlotResult, delta_plot


def _data(seed=0, n_persons=120, n_items=8):
    rng = np.random.default_rng(seed)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(float)
    g = np.array([0] * (n_persons // 2) + [1] * (n_persons - n_persons // 2))
    return y, g


def test_delta_plot_happy_norm_constraint():
    y, g = _data()
    res = delta_plot(y, g)
    assert isinstance(res, DeltaPlotResult)
    assert res.props.shape == (8, 2)
    assert res.deltas.shape == (8, 2)


def test_delta_plot_fixed_threshold():
    y, g = _data()
    assert isinstance(delta_plot(y, g, threshold="fixed"), DeltaPlotResult)


def test_delta_plot_extreme_add():
    y, g = _data()
    assert isinstance(delta_plot(y, g, extreme="add", nr_add=1), DeltaPlotResult)


def test_delta_plot_purify_variant():
    y, g = _data()
    assert isinstance(delta_plot(y, g, purify="IPP1", max_iter=3), DeltaPlotResult)


def test_delta_plot_accepts_bool_dtype():
    y, g = _data()
    assert isinstance(delta_plot(y.astype(bool), g), DeltaPlotResult)


def test_delta_plot_requires_rust_core():
    y, g = _data()
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            delta_plot(y, g)


def test_delta_plot_rejects_bad_response_shape():
    with pytest.raises(ValueError):
        delta_plot(np.zeros(8), np.array([0, 1]))
    with pytest.raises(ValueError):
        delta_plot(np.zeros((0, 2)), np.array([]))
    with pytest.raises(ValueError):
        delta_plot(np.zeros((5, 1)), np.zeros(5))


def test_delta_plot_rejects_complex_responses():
    with pytest.raises(ValueError):
        delta_plot(np.zeros((4, 2), dtype=complex), np.array([0, 1, 0, 1]))


def test_delta_plot_rejects_non_numeric_responses():
    with pytest.raises(ValueError):
        delta_plot(np.array([["a", "b"], ["c", "d"]]), np.array([0, 1]))


def test_delta_plot_rejects_non_binary_values():
    y, g = _data()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        delta_plot(y, g)


def test_delta_plot_rejects_bad_group():
    y, _ = _data()
    with pytest.raises(ValueError):
        delta_plot(y, np.zeros((120, 1)))  # not 1-D
    with pytest.raises(ValueError):
        delta_plot(y, np.zeros(10))  # wrong length
    with pytest.raises(ValueError):
        delta_plot(y, np.full(120, 1j))  # complex
    with pytest.raises(ValueError):
        delta_plot(y, np.full(120, 2))  # not 0/1


def test_delta_plot_rejects_bad_options():
    y, g = _data()
    with pytest.raises(ValueError):
        delta_plot(y, g, threshold="bogus")
    with pytest.raises(ValueError):
        delta_plot(y, g, extreme="bogus")
    with pytest.raises(ValueError):
        delta_plot(y, g, purify="IPP9")
    with pytest.raises(ValueError):
        delta_plot(y, g, extreme="add", nr_add=0)
