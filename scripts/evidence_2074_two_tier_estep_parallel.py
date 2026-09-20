#!/usr/bin/env python3
"""Small-study evidence for #2074 two-tier E-step person-chunk parallelism.

NOT A4 / NOT q=241 n=340. Reports wall time, max RSS proxy via resource,
and bit-identity of parameters across e_step_n_threads at fixed chunks.
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from fast_mlsirm.two_tier_grm import fit_two_tier_grm

N_PERSONS = 48
N_ITEMS = 8
N_CAT = 3
N_PRIMARY = 2
N_SPECIFIC = 2
Q = 7
MAX_ITER = 12
SEED = 0xC0FFEE
CHUNKS = 4

PRIMARY_MAP = np.array(
    [
        [True, False],
        [True, False],
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ],
    dtype=bool,
)
SPECIFIC_MAP = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)


def _simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(N_PERSONS, N_PRIMARY))
    y = np.zeros((N_PERSONS, N_ITEMS), dtype=np.int64)
    a_p = np.array([1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 0.95, 1.05])
    a_s = np.array([0.7, 0.6, 0.65, 0.75, 0.55, 0.7, 0.6, 0.8])
    d = np.array([[1.0, -0.5], [0.8, -0.7], [1.1, -0.4], [0.9, -0.6]] * 2)
    for i in range(N_ITEMS):
        p_idx = 0 if PRIMARY_MAP[i, 0] else 1
        s_idx = SPECIFIC_MAP[i]
        # Approximate specific draw shared within block for simulation only.
        eta = a_p[i] * theta[:, p_idx] + a_s[i] * (0.3 * (s_idx + 1))
        cum = 1.0 / (1.0 + np.exp(-(eta[:, None] + d[i])))
        probs = np.column_stack(
            [1.0 - cum[:, 0], cum[:, 0] - cum[:, 1], cum[:, 1]]
        )
        draws = rng.random(N_PERSONS)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y


def _run(y: np.ndarray, threads: int):
    t0 = time.perf_counter()
    fit = fit_two_tier_grm(
        y,
        PRIMARY_MAP,
        SPECIFIC_MAP,
        N_CAT,
        N_PRIMARY,
        N_SPECIFIC,
        Q,
        Q,
        MAX_ITER,
        1e-5,
        1,
        SEED,
        e_step_n_chunks=CHUNKS,
        e_step_n_threads=threads,
    )
    wall = time.perf_counter() - t0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return fit, wall, ru.ru_utime + ru.ru_stime, ru.ru_maxrss


def main() -> None:
    out_dir = Path(
        os.environ.get(
            "FMLS_2074_EVIDENCE_DIR",
            "docs/orchestration/evidence-2074",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    y = _simulate(SEED)

    # Baseline: single-thread floor (two repeats).
    f1a, wall1a, cpu1a, rss1a = _run(y, 1)
    f1b, wall1b, cpu1b, rss1b = _run(y, 1)
    floor_bits = {
        "a_primary_equal": bool(np.array_equal(f1a.a_primary, f1b.a_primary)),
        "threshold_equal": bool(np.array_equal(f1a.threshold, f1b.threshold)),
        "phi_equal": bool(np.array_equal(f1a.phi, f1b.phi)),
        "loglik_equal": bool(np.array_equal(f1a.loglik_trace, f1b.loglik_trace)),
    }

    n_threads = max(2, min(os.cpu_count() or 2, 4))
    f_mt, wall_mt, cpu_mt, rss_mt = _run(y, n_threads)
    equiv = {
        "a_primary_equal": bool(np.array_equal(f1a.a_primary, f_mt.a_primary)),
        "threshold_equal": bool(np.array_equal(f1a.threshold, f_mt.threshold)),
        "phi_equal": bool(np.array_equal(f1a.phi, f_mt.phi)),
        "loglik_equal": bool(np.array_equal(f1a.loglik_trace, f_mt.loglik_trace)),
    }

    report = {
        "setting": {
            "n_persons": N_PERSONS,
            "n_items": N_ITEMS,
            "q": Q,
            "max_iter": MAX_ITER,
            "e_step_n_chunks": CHUNKS,
            "note": "small study; NOT A4; NOT q=241 n=340",
        },
        "baseline_threads_1": {
            "wall_s_run_a": wall1a,
            "wall_s_run_b": wall1b,
            "cpu_s_run_a": cpu1a,
            "maxrss_run_a": rss1a,
            "bit_identical_repeats": floor_bits,
        },
        "parallel": {
            "e_step_n_threads": n_threads,
            "wall_s": wall_mt,
            "cpu_s": cpu_mt,
            "maxrss": rss_mt,
            "wall_speedup_vs_baseline_a": (wall1a / wall_mt) if wall_mt > 0 else None,
            "cpu_over_wall": (cpu_mt / wall_mt) if wall_mt > 0 else None,
            "bit_identical_vs_threads_1": equiv,
        },
    }
    out_path = out_dir / "two_tier_estep_parallel_evidence.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert all(floor_bits.values()), "n_jobs=1 floor must be bit-identical"
    assert all(equiv.values()), "fixed chunks must stay bit-identical across threads"


if __name__ == "__main__":
    main()
