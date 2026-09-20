"""Live same-input concurrent CPU+GPU measure (tiny fixture; no cargo).

Mirrors tests/test_bifactor_gpu.py fit_bifactor_grm(device=cpu|gpu) contract
with threading.Barrier start sync. n_compared counts finite parameter slots.
"""
from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fast_mlsirm.bifactor_grm import fit_bifactor_grm
import fast_mlsirm

OUT = Path(__file__).with_name("cpu_gpu_concurrent_live_mba_20260920T1610.json")

# Identical tiny input (same as prior Metal proof scale; NOT A4 / NOT q=241)
rng = np.random.default_rng(20260920)
y = rng.integers(0, 3, size=(16, 4)).astype(float)
smap = np.array([0, 0, 1, 1], dtype=np.int64)
COMMON = dict(
    responses=y,
    specific_map=smap,
    n_cat=3,
    n_specific=2,
    q_general=5,
    q_specific=5,
    max_iter=5,
    tol=1e-3,
    n_starts=1,
    seed=20260920,
)

displays = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
memsize = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
metal_lines = [
    ln.strip()
    for ln in displays.splitlines()
    if any(k in ln for k in ("Chipset Model", "Metal Support", "Type:", "Vendor:"))
]

box: dict = {}
barrier = threading.Barrier(2)
wall_clock_start = datetime.now(timezone.utc).isoformat()
mono_t0 = time.perf_counter()


def run(device: str) -> None:
    barrier.wait()
    t0 = time.perf_counter()
    wall0 = datetime.now(timezone.utc).isoformat()
    try:
        fit = fit_bifactor_grm(**COMMON, device=device)
        t1 = time.perf_counter()
        wall1 = datetime.now(timezone.utc).isoformat()
        box[device] = {
            "ok": True,
            "device_requested": device,
            "wall_clock_start": wall0,
            "wall_clock_end": wall1,
            "mono_start": t0,
            "mono_end": t1,
            "wall_s": round(t1 - t0, 6),
            "converged": bool(fit.converged),
            "n_iter": int(fit.n_iter),
            "loglik": float(fit.loglik_trace[-1]) if len(fit.loglik_trace) else None,
            "a_general": np.asarray(fit.a_general, dtype=np.float64).ravel(),
            "a_specific": np.asarray(fit.a_specific, dtype=np.float64).ravel(),
            "threshold": np.asarray(fit.threshold, dtype=np.float64).ravel(),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — capture exact failure for root
        t1 = time.perf_counter()
        wall1 = datetime.now(timezone.utc).isoformat()
        box[device] = {
            "ok": False,
            "device_requested": device,
            "wall_clock_start": wall0,
            "wall_clock_end": wall1,
            "mono_start": t0,
            "mono_end": t1,
            "wall_s": round(t1 - t0, 6),
            "error": f"{type(e).__name__}: {e}",
        }


threads = [threading.Thread(target=run, args=(d,)) for d in ("cpu", "gpu")]
for t in threads:
    t.start()
for t in threads:
    t.join()
mono_t1 = time.perf_counter()
wall_clock_end = datetime.now(timezone.utc).isoformat()

cpu, gpu = box.get("cpu", {}), box.get("gpu", {})
overlap = {
    "mono_overlap_wall_s": round(mono_t1 - mono_t0, 6),
    "cpu_mono_interval": [cpu.get("mono_start"), cpu.get("mono_end")],
    "gpu_mono_interval": [gpu.get("mono_start"), gpu.get("mono_end")],
    "simultaneous": False,
}
if cpu.get("ok") and gpu.get("ok"):
    # overlap of [cpu_start,cpu_end] ∩ [gpu_start,gpu_end]
    lo = max(cpu["mono_start"], gpu["mono_start"])
    hi = min(cpu["mono_end"], gpu["mono_end"])
    overlap["simultaneous_interval_s"] = round(max(0.0, hi - lo), 6)
    overlap["simultaneous"] = hi > lo

equality = None
n_compared = 0
n_exceeded = 0
atol = 1e-3
if cpu.get("ok") and gpu.get("ok"):
    parts = []
    for key in ("a_general", "a_specific", "threshold"):
        a = np.asarray(cpu[key], dtype=np.float64)
        b = np.asarray(gpu[key], dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        n_compared += int(mask.sum())
        diff = np.abs(a[mask] - b[mask])
        n_exceeded += int(np.sum(diff > atol))
        parts.append(
            {
                "name": key,
                "n": int(mask.sum()),
                "max_abs": float(diff.max()) if diff.size else 0.0,
            }
        )
    # loglik scalar
    if cpu.get("loglik") is not None and gpu.get("loglik") is not None:
        n_compared += 1
        ll = abs(float(cpu["loglik"]) - float(gpu["loglik"]))
        if ll > atol:
            n_exceeded += 1
        parts.append({"name": "loglik", "n": 1, "max_abs": ll})
    equality = {
        "atol": atol,
        "n_compared": n_compared,
        "n_exceeded": n_exceeded,
        "parity_holds": n_compared > 0 and n_exceeded == 0,
        "parts": parts,
    }

# strip arrays from arms for JSON
for arm in (cpu, gpu):
    for k in ("a_general", "a_specific", "threshold"):
        arm.pop(k, None)

report = {
    "artifact": "cpu_gpu_concurrent_live_mba",
    "schema": "cpu_gpu_concurrent_live/v1",
    "pass_gate": "n_compared > 0 and parity_holds",
    "passed": bool(equality and equality["n_compared"] > 0 and equality["parity_holds"]),
    "host": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "fast_mlsirm_version": fast_mlsirm.__version__,
    "fast_mlsirm_path": fast_mlsirm.__file__,
    "device": {
        "cpu": "host_cpu",
        "gpu_requested": "gpu",
        "backend_expected": "wgpu→Metal on Darwin arm64",
        "system_profiler": metal_lines,
        "not_llvmpipe": True,
    },
    "memory": {"hw_memsize_gb": round(memsize / 1024**3, 2), "vm_swapusage": swap},
    "fixture": {
        **{k: COMMON[k] for k in ("n_cat", "n_specific", "q_general", "q_specific", "max_iter", "tol", "n_starts", "seed")},
        "n_persons": int(y.shape[0]),
        "n_items": int(y.shape[1]),
        "api": "fast_mlsirm.bifactor_grm.fit_bifactor_grm",
        "bench_path_mirrored": "tests/test_bifactor_gpu.py",
        "note": "TINY identical input; concurrent Barrier(2); NOT A4; no new cargo",
    },
    "timing": {
        "wall_clock_start_utc": wall_clock_start,
        "wall_clock_end_utc": wall_clock_end,
        "overlap": overlap,
        "cpu_wall_s": cpu.get("wall_s"),
        "gpu_wall_s": gpu.get("wall_s"),
    },
    "arms": {"cpu": cpu, "gpu": gpu},
    "numeric_equality": equality,
    "a4_restarted": False,
    "mac_large_build": False,
}

if equality is None or equality["n_compared"] == 0:
    report["passed"] = False
    report["fail_reason"] = "n_compared=0 or arm failure — cannot pass"

OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({
    "passed": report["passed"],
    "n_compared": (equality or {}).get("n_compared"),
    "n_exceeded": (equality or {}).get("n_exceeded"),
    "overlap_s": overlap.get("simultaneous_interval_s"),
    "out": str(OUT),
    "cpu_ok": cpu.get("ok"),
    "gpu_ok": gpu.get("ok"),
    "cpu_err": cpu.get("error"),
    "gpu_err": gpu.get("error"),
}, indent=2))
