"""Expected-raw person scoring for fitted two-tier GRM models."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.two_tier_grm import expected_raw_two_tier_grm, fit_two_tier_grm

N_PERSONS = 80
N_ITEMS = 6
N_CAT = 3
N_PRIMARY = 2
N_SPECIFIC = 2
SEED = 0xC0FFEE


PRIMARY_MAP = np.array(
    [
        [True, False],
        [True, False],
        [True, True],
        [True, True],
        [True, False],
        [True, False],
    ],
    dtype=bool,
)
SPECIFIC_MAP = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)


def _simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CAT, size=(N_PERSONS, N_ITEMS), dtype=np.int64)


def _fit(responses: np.ndarray):
    return fit_two_tier_grm(
        responses,
        PRIMARY_MAP,
        SPECIFIC_MAP,
        N_CAT,
        N_PRIMARY,
        N_SPECIFIC,
        q_primary=21,
        q_specific=21,
        max_iter=300,
        tol=1e-5,
        n_starts=1,
        seed=SEED,
    )


def test_expected_raw_two_tier_grm_smoke() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    scores = expected_raw_two_tier_grm(fit, SPECIFIC_MAP, q_specific=21)
    assert scores.shape == (N_PERSONS,)
    assert np.all(np.isfinite(scores))
    lo = 0.0
    hi = float(N_ITEMS * (N_CAT - 1))
    assert float(scores.min()) >= lo - 1e-9
    assert float(scores.max()) <= hi + 1e-9


def test_expected_raw_requires_q_specific() -> None:
    y = _simulate(SEED + 1)
    fit = _fit(y)
    with pytest.raises(TypeError):
        expected_raw_two_tier_grm(fit, SPECIFIC_MAP)  # type: ignore[call-arg]
