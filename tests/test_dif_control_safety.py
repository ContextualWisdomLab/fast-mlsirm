"""Public regressions for observed-score DIF semantic-control admission."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import fast_mlsirm as fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import dif


class _HostileArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("caller data materialized before control rejection")


class _HostileFloat(float):
    def __float__(self):
        raise AssertionError("caller float callback executed")


class _HostileInt(int):
    def __int__(self):
        raise AssertionError("caller integer callback executed")


class _TruthProvider:
    def __bool__(self):
        raise AssertionError("caller truth callback executed")


class _FloatProvider:
    def __float__(self):
        raise AssertionError("caller float protocol executed")


class _IndexProvider:
    def __index__(self):
        raise AssertionError("caller index protocol executed")


def _forbid_core(monkeypatch):
    def _unexpected_core():
        raise AssertionError("compiled core discovered before control rejection")

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"exclude_studied_item": _TruthProvider()}, "exclude_studied_item"),
        ({"fdr_q": _HostileFloat(0.05)}, "fdr_q"),
        ({"fdr_q": _FloatProvider()}, "fdr_q"),
        ({"fdr_q": True}, "fdr_q"),
        ({"max_iter": _HostileInt(50)}, "max_iter"),
        ({"max_iter": _IndexProvider()}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"fdr_q": 0.0}, "fdr_q"),
        ({"fdr_q": float("nan")}, "fdr_q"),
        ({"max_iter": -1}, "max_iter"),
    ],
)
def test_logistic_dif_rejects_invalid_controls_before_data_or_core(monkeypatch, kwargs, error):
    _forbid_core(monkeypatch)
    with pytest.raises((TypeError, ValueError), match=error):
        dif.logistic_dif(_HostileArray(), _HostileArray(), **kwargs)


@pytest.mark.parametrize(
    ("func_name", "kwargs", "error"),
    [
        ("mantel_haenszel_dif_purified", {"exclude_studied_item": _TruthProvider()}, "exclude_studied_item"),
        ("mantel_haenszel_dif_purified", {"fdr_q": _HostileFloat(0.05)}, "fdr_q"),
        ("mantel_haenszel_dif_purified", {"max_rounds": _HostileInt(3)}, "max_rounds"),
        ("mantel_haenszel_dif_purified", {"min_anchor_items": _IndexProvider()}, "min_anchor_items"),
        ("mantel_haenszel_dif_purified", {"max_rounds": 0}, "max_rounds"),
        ("mantel_haenszel_dif_purified", {"min_anchor_items": -1}, "min_anchor_items"),
        ("logistic_dif_purified", {"max_iter": _HostileInt(50)}, "max_iter"),
        ("logistic_dif_purified", {"max_iter": True}, "max_iter"),
        ("logistic_dif_purified", {"max_rounds": 0}, "max_rounds"),
    ],
)
def test_purified_dif_rejects_invalid_controls_before_data_or_core(
    monkeypatch, func_name, kwargs, error
):
    _forbid_core(monkeypatch)
    func = getattr(dif, func_name)
    with pytest.raises((TypeError, ValueError), match=error):
        func(_HostileArray(), _HostileArray(), **kwargs)


@dataclass
class _CaptureCore:
    logistic_controls: tuple[object, ...] | None = None
    mh_purified_controls: tuple[object, ...] | None = None
    logistic_purified_controls: tuple[object, ...] | None = None

    @staticmethod
    def _logistic_result():
        return {
            "item": [0],
            "chi2_uniform": [0.0],
            "p_uniform": [1.0],
            "chi2_nonuniform": [0.0],
            "p_nonuniform": [1.0],
            "chi2_total": [0.0],
            "p_total": [1.0],
            "delta_r2": [0.0],
            "delta_r2_uniform": [0.0],
            "jg_class": ["A"],
            "flagged_bh": [False],
            "converged": [True],
        }

    @staticmethod
    def _mh_result():
        return {
            "item": [0],
            "alpha_mh": [1.0],
            "chi2_mh": [0.0],
            "p_value": [1.0],
            "mh_d_dif": [0.0],
            "se_d_dif": [0.0],
            "std_p_dif": [0.0],
            "ets_class": ["A"],
            "flagged_bh": [False],
        }

    @staticmethod
    def _purify_meta(result):
        return result | {
            "anchor": [True],
            "n_anchor": 1,
            "rounds": 0,
            "purify_converged": True,
            "purify_termination_reason": "stable_flag_set",
        }

    def logistic_dif(self, yy, gg, n_persons, n_items, exclude, fdr_q, max_iter):
        self.logistic_controls = (exclude, fdr_q, max_iter)
        return self._logistic_result()

    def mantel_haenszel_dif_purified(
        self, yy, gg, n_persons, n_items, exclude, fdr_q, max_rounds, min_anchor_items
    ):
        self.mh_purified_controls = (exclude, fdr_q, max_rounds, min_anchor_items)
        return self._purify_meta(self._mh_result())

    def logistic_dif_purified(
        self, yy, gg, n_persons, n_items, exclude, fdr_q, max_iter, max_rounds, min_anchor_items
    ):
        self.logistic_purified_controls = (
            exclude,
            fdr_q,
            max_iter,
            max_rounds,
            min_anchor_items,
        )
        return self._purify_meta(self._logistic_result())


def test_public_dif_controls_normalize_supported_numpy_scalars(monkeypatch):
    core = _CaptureCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses = np.array([[0], [1]], dtype=np.int8)
    group = np.array([0, 1], dtype=np.int8)

    dif.logistic_dif(
        responses,
        group,
        exclude_studied_item=np.bool_(False),
        fdr_q=np.float32(0.05),
        max_iter=np.int16(3),
    )
    dif.mantel_haenszel_dif_purified(
        responses,
        group,
        exclude_studied_item=np.bool_(False),
        fdr_q=np.float32(0.05),
        max_rounds=np.uint8(2),
        min_anchor_items=np.int16(0),
    )
    dif.logistic_dif_purified(
        responses,
        group,
        exclude_studied_item=np.bool_(False),
        fdr_q=np.float32(0.05),
        max_iter=np.int16(3),
        max_rounds=np.uint8(2),
        min_anchor_items=np.int16(0),
    )

    for value in core.logistic_controls:
        assert type(value) in (bool, float, int)
    for value in core.mh_purified_controls:
        assert type(value) in (bool, float, int)
    for value in core.logistic_purified_controls:
        assert type(value) in (bool, float, int)


def test_package_aliases_share_hardened_dif_functions():
    assert fast_mlsirm.logistic_dif is dif.logistic_dif
    assert fast_mlsirm.mantel_haenszel_dif_purified is dif.mantel_haenszel_dif_purified
    assert fast_mlsirm.logistic_dif_purified is dif.logistic_dif_purified
