"""Fail-closed core and numeric-domain coverage for reliability wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.reliability as reliability


@pytest.mark.parametrize(
    "call",
    [
        lambda: reliability.icc(np.zeros((2, 2))),
        lambda: reliability.kripp_alpha(np.zeros((2, 2))),
        lambda: reliability.finn_coefficient(np.zeros((2, 2)), 2),
        lambda: reliability.maxwell_re(np.zeros((2, 2))),
        lambda: reliability.robinson_a(np.zeros((2, 2))),
        lambda: reliability.mean_pairwise_cor(np.zeros((2, 2))),
        lambda: reliability.mean_pairwise_rho(np.zeros((2, 2))),
        lambda: reliability.stuart_maxwell_mh(np.zeros((2, 2))),
        lambda: reliability.bhapkar_mh(np.zeros((2, 2))),
        lambda: reliability.rater_bias(np.zeros((2, 2))),
        lambda: reliability.n_cohen_kappa(0.4, 0.5, 0.3, 0.1),
    ],
)
def test_reliability_wrappers_require_the_rust_core(monkeypatch, call) -> None:
    """Every public reliability statistic fails closed when its Rust owner is absent."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="compiled Rust core"):
        call()


@pytest.mark.parametrize(
    ("call", "core_attribute"),
    [
        (lambda: reliability.icc(np.array([["a", "b"]])), "icc"),
        (lambda: reliability.kripp_alpha(np.array([["a", "b"]])), "kripp_alpha"),
        (lambda: reliability.finn_coefficient(np.array([["a", "b"]]), 2), "finn_coefficient"),
        (lambda: reliability.maxwell_re(np.array([["a", "b"]])), "maxwell_re"),
        (lambda: reliability.robinson_a(np.array([["a", "b"]])), "robinson_a"),
        (lambda: reliability.mean_pairwise_cor(np.array([["a", "b"]])), "mean_pairwise_cor"),
        (lambda: reliability.mean_pairwise_rho(np.array([["a", "b"]])), "mean_pairwise_rho"),
        (lambda: reliability.stuart_maxwell_mh(np.array([["a", "b"], ["c", "d"]])), "stuart_maxwell_mh"),
        (lambda: reliability.bhapkar_mh(np.array([["a", "b"], ["c", "d"]])), "bhapkar_mh"),
        (lambda: reliability.rater_bias(np.array([["a", "b"], ["c", "d"]])), "rater_bias"),
    ],
)
def test_reliability_wrappers_reject_non_numeric_arrays_before_ffi(
    monkeypatch, call, core_attribute
) -> None:
    """Text arrays fail at the Python contract before any native method executes."""
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(**{core_attribute: object()}),
    )
    with pytest.raises(ValueError, match="numeric"):
        call()


def test_n_cohen_kappa_rejects_complex_scalars_before_ffi(monkeypatch) -> None:
    """Sample-size inputs remain real-valued at the trust boundary."""
    monkeypatch.setattr(
        fitstats,
        "_core_module",
        lambda: SimpleNamespace(n_cohen_kappa=object()),
    )
    with pytest.raises(ValueError, match="real-valued"):
        reliability.n_cohen_kappa(0.4, complex(0.5, 1.0), 0.3, 0.1)
