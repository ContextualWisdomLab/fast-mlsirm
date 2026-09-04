"""Regression contracts for local-dependence public control admission."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.fitstats as fitstats_module


class _BombInt(int):
    """Integer subclass that proves public validation never executes coercion hooks."""

    def __int__(self):
        raise AssertionError("caller-controlled __int__ executed")


class _BombCore:
    """Native stand-in that fails if invalid controls reach Rust dispatch."""

    def ld_indices(self, *_args, **_kwargs):
        raise AssertionError("invalid quadrature control reached the native core")


def _fixture():
    responses = np.zeros((20, 2), dtype=np.float64)
    factor_id = np.zeros(2, dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(2),
        b=np.zeros(2),
        zeta=np.zeros((2, 1)),
        tau=-30.0,
    )
    return responses, factor_id, params


def test_ld_indices_rejects_unsupported_theta_quadrature_before_native(monkeypatch):
    """The always-used trait grid must be an embedded Gauss-Hermite rule."""
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _BombCore())
    responses, factor_id, params = _fixture()

    with pytest.raises(ValueError, match="q_theta must be one of"):
        fitstats_module.ld_indices(
            responses, factor_id, params, "MIRT", q_theta=3, q_xi=3
        )


def test_ld_indices_rejects_integer_subclasses_without_callbacks(monkeypatch):
    """Callback-bearing integer subclasses fail before coercion or native discovery."""
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _BombCore())
    responses, factor_id, params = _fixture()

    with pytest.raises(ValueError, match="q_theta must be a positive integer"):
        fitstats_module.ld_indices(
            responses, factor_id, params, "MIRT", q_theta=_BombInt(7), q_xi=3
        )
    with pytest.raises(ValueError, match="q_xi must be a positive integer"):
        fitstats_module.ld_indices(
            responses, factor_id, params, "MIRT", q_theta=7, q_xi=_BombInt(3)
        )


def test_ld_indices_package_root_uses_the_hardened_callable():
    """The package-root export must share the same hardened public boundary."""
    assert fast_mlsirm.ld_indices is fitstats_module.ld_indices
