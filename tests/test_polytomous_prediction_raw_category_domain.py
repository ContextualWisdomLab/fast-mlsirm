"""Regression for raw polytomous prediction category-domain replay."""

from __future__ import annotations

import subprocess
import sys


def test_raw_prediction_replays_category_ceiling_before_core_discovery() -> None:
    """The private raw helper must reject 65 categories before Rust discovery."""

    script = r'''
import importlib
import numpy as np
import fast_mlsirm.polytomous as poly

poly = importlib.reload(poly)


def core_forbidden():
    raise AssertionError("compiled core discovery must not occur")


poly._core_module = core_forbidden
fit = poly.PolytomousFit(
    model="grm",
    slope=np.ones(1, dtype=np.float64),
    cat_params=np.zeros((1, 64), dtype=np.float64),
    loglik=0.0,
    n_iter=1,
)

try:
    poly._polytomous_predictions(fit, np.array([0.0], dtype=np.float64))
except ValueError as exc:
    assert str(exc) == "n_cat must be in 2..=64", str(exc)
else:
    raise AssertionError("expected category-domain ValueError")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
