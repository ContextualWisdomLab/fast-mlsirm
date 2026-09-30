# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""``fit_poly_fipc`` must release the GIL while the Rust EM runs (#2001).

Wall-clock speedup ratios drift with host load, so this gate measures the GIL
directly. While a worker thread runs one FIPC fit, the main thread records a
scheduling gap between bounded loop samples. A binding that holds the GIL freezes the main
thread for the whole fit, which leaves a gap about as long as the fit itself.
A detached binding lets the main thread keep ticking, so the largest gap stays
near the interpreter switch interval (``sys.getswitchinterval()``, 5 ms by
default). The bound is half the measured fit duration, and the fixture keeps
the fit far longer than the switch interval, so the two cases separate by
construction instead of by a tuned constant.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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


def _observe_fit(fit, *, timeout_seconds: float = 60.) -> tuple[float, float]:
    """Observe scheduling gaps with constant-size state and a bounded deadline.

    The calling process must also have an external timeout: Python cannot
    interrupt a native call that never releases the GIL. The numerical test
    below runs in an owned child process for precisely that reason.
    """
    done = threading.Event()
    bounds: list[float] = []
    errors: list[BaseException] = []

    def worker() -> None:
        bounds.append(time.perf_counter())
        try:
            fit()
        except BaseException as error:
            errors.append(error)
        finally:
            bounds.append(time.perf_counter())
            done.set()

    deadline = time.perf_counter() + timeout_seconds
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    previous = None
    largest_gap = 0.
    while not done.is_set():
        now = time.perf_counter()
        if now >= deadline:
            raise TimeoutError("GIL observer deadline exceeded")
        if bounds and now >= bounds[0] and (len(bounds) == 1 or now <= bounds[1]):
            previous = bounds[0] if previous is None else previous
            largest_gap = max(largest_gap, now - previous)
            previous = now
        done.wait(.001)  # bounded sampling; no busy-spin or timestamp list
    thread.join(timeout=max(0., deadline - time.perf_counter()))
    if thread.is_alive():
        raise TimeoutError("GIL observer deadline exceeded while joining")
    if errors:
        raise errors[0]
    start, end = bounds
    if end > deadline:
        raise TimeoutError("GIL observer deadline exceeded before completion")
    previous = start if previous is None else previous
    largest_gap = max(largest_gap, end - previous)
    return end - start, largest_gap


def _run_numeric_probe() -> None:
    import fast_mlsirm._core as core
    extension = Path(core.__file__).resolve()
    with extension.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(json.dumps({"interpreter": sys.executable, "extension": str(extension),
                      "extension_sha256": digest}), flush=True)
    data = _fixture()
    _fit(data)  # warm-up is also inside the owned process's external timeout
    fit_seconds, largest_gap = _observe_fit(lambda: _fit(data))
    assert fit_seconds > 0.05, f"fixture too fast to separate GIL states ({fit_seconds:.3f}s)"
    assert largest_gap < fit_seconds / 2, (
        f"main thread froze for {largest_gap:.3f}s of a {fit_seconds:.3f}s fit; "
        "fit_poly_fipc is holding the GIL (missing py.detach)"
    )


def test_fit_poly_fipc_lets_other_python_threads_run() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root / "python"), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--numeric-probe"],
        cwd=root, env=env, capture_output=True, text=True, timeout=90.,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if sys.argv[1:] != ["--numeric-probe"]:
        raise SystemExit("expected --numeric-probe")
    _run_numeric_probe()
