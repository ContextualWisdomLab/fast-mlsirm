"""Fail-closed contracts for local-dependence model scope and workspace."""

from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats_module


class _BombCore:
    """Native stand-in that proves invalid LD requests stop before dispatch."""

    def ld_indices(self, *_args, **_kwargs):
        raise AssertionError("invalid local-dependence request reached the native core")


def _params(n_items: int, latent_dim: int = 1) -> SimpleNamespace:
    """Build inert item parameters for public-boundary contract tests."""
    return SimpleNamespace(
        alpha=np.zeros(n_items),
        b=np.zeros(n_items),
        zeta=np.zeros((n_items, latent_dim)),
        tau=-30.0,
    )


def test_ld_indices_rejects_multidimensional_trait_bank_before_native(monkeypatch):
    """Current LD quadrature must not alias independent trait dimensions."""
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _BombCore())

    with pytest.raises(ValueError, match="one trait dimension"):
        fitstats_module.ld_indices(
            np.zeros((20, 2)),
            np.array([0, 1], dtype=np.int64),
            _params(2),
            "MIRT",
            q_theta=7,
            q_xi=3,
        )


def test_ld_indices_rejects_probability_workspace_bomb_before_native(monkeypatch):
    """A small request surface cannot authorize multi-gigabyte ICC allocation."""
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _BombCore())
    n_items = 1_000

    with pytest.raises(ValueError, match="workspace"):
        fitstats_module.ld_indices(
            np.zeros((20, n_items)),
            np.zeros(n_items, dtype=np.int64),
            _params(n_items, latent_dim=3),
            "LSIRM",
            q_theta=41,
            q_xi=41,
        )


def test_ld_indices_rejects_pair_person_work_bomb_before_native(monkeypatch):
    """Quadratic pair-by-person work is bounded independently of ICC storage."""
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _BombCore())
    n_items = 2_000

    with pytest.raises(ValueError, match="pair-person work"):
        fitstats_module.ld_indices(
            np.zeros((200, n_items)),
            np.zeros(n_items, dtype=np.int64),
            _params(n_items),
            "MIRT",
            q_theta=7,
            q_xi=3,
        )


def test_ld_indices_surfaces_and_rejects_unsupported_population_before_native(monkeypatch):
    """Non-single fitted populations cannot silently use standard-normal expectations."""
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: _BombCore())
    assert "population" in signature(fitstats_module.ld_indices).parameters

    with pytest.raises(ValueError, match="population.*single"):
        fitstats_module.ld_indices(
            np.zeros((20, 2)),
            np.zeros(2, dtype=np.int64),
            _params(2),
            "MIRT",
            q_theta=7,
            q_xi=3,
            population={"kind": "singlefree"},
        )
