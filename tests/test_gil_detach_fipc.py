# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""``fit_poly_fipc`` must release the GIL while the Rust EM runs (#2001).

Wall-clock speedup ratios drift with host load, so this gate measures the GIL
directly. While a worker thread runs one FIPC fit, the main thread records a
timestamp on every loop pass. A binding that holds the GIL freezes the main
thread for the whole fit, which leaves a gap about as long as the fit itself.
A detached binding lets the main thread keep ticking, so the largest gap stays
near the interpreter switch interval (``sys.getswitchinterval()``, 5 ms by
default). The bound is half the measured fit duration, and the fixture keeps
the fit far longer than the switch interval, so the two cases separate by
construction instead of by a tuned constant.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from fast_mlsirm.polytomous import fit_poly_fipc, fit_polytomous

N_PERSONS, N_ITEMS, N_CAT = 1500, 12, 4


def _fixture() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    theta = rng.standard_normal(N_PERSONS)
    slope = rng.uniform(0.8, 1.6, N_ITEMS)
    cuts = np.sort(rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1)), axis=1)
    eta = slope[None, :, None] * (theta[:, None, None] - cuts[None, :, :])
    y = (rng.random((N_PERSONS, N_ITEMS, 1)) < 1.0 / (1.0 + np.exp(-eta))).sum(axis=2)
    ref = fit_polytomous(y[:, : N_ITEMS // 2], n_cat=N_CAT, model="grm", q_theta=21, max_iter=50, tol=1e-4)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[: N_ITEMS // 2] = True
    anchor_slope = np.zeros(N_ITEMS)
    anchor_slope[: N_ITEMS // 2] = ref.slope
    anchor_cat = np.zeros((N_ITEMS, N_CAT - 1))
    anchor_cat[: N_ITEMS // 2] = ref.cat_params
    return {
        "responses": y.astype(np.int64),
        "anchor": anchor,
        "anchor_slope": anchor_slope,
        "anchor_cat_params": anchor_cat,
    }


def _fit(data: dict[str, object]) -> None:
    fit_poly_fipc(
        data["responses"],
        N_CAT,
        data["anchor"],
        data["anchor_slope"],
        data["anchor_cat_params"],
        q_theta=121,
        max_iter=60,
        tol=1e-12,
    )


def test_fit_poly_fipc_lets_other_python_threads_run() -> None:
    data = _fixture()
    _fit(data)  # warm-up: exclude first-call costs

    done = threading.Event()
    bounds: list[float] = []

    def worker() -> None:
        bounds.append(time.perf_counter())
        _fit(data)
        bounds.append(time.perf_counter())
        done.set()

    ticks: list[float] = []
    thread = threading.Thread(target=worker)
    thread.start()
    while not done.is_set():
        ticks.append(time.perf_counter())
    thread.join()

    start, end = bounds
    fit_seconds = end - start
    assert fit_seconds > 0.05, f"fixture too fast to separate GIL states ({fit_seconds:.3f}s)"
    inside = [t for t in ticks if start <= t <= end]
    gaps = np.diff([start, *inside, end])
    assert gaps.max() < fit_seconds / 2, (
        f"main thread froze for {gaps.max():.3f}s of a {fit_seconds:.3f}s fit; "
        "fit_poly_fipc is holding the GIL (missing py.detach)"
    )
