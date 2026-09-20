"""Live same-input concurrent CPU+GPU measure (tiny fixture; no cargo).

Mirrors tests/test_bifactor_gpu.py fit_bifactor_grm(device=cpu|gpu) contract
with threading.Barrier start sync. n_compared counts finite parameter slots.

Reinforced evidence contract (root review of 2a8b7f0b):
- Record the *selected* display/GPU adapter from system_profiler (not a
  hardcoded not_llvmpipe flag).
- Full shape / finite / count checks on compared arrays.
- Separate overlap verdict; rename span timing away from mono_overlap_wall_s.
- Explicit: call-time interval overlap ≠ hardware kernel concurrency.
- converged=false ⇒ research/speed claims disallowed (smoke only).
"""
from __future__ import annotations

import json
import platform
import re
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


def _parse_selected_adapter(displays: str) -> dict:
    """Derive the selected GPU adapter from SPDisplaysDataType text.

    Prefers the first Type: GPU block. ``not_llvmpipe`` is computed from the
    selected name/vendor/Metal line — never hardcoded True.
    """
    lines = [ln.rstrip() for ln in displays.splitlines()]
    blocks: list[dict] = []
    current: dict | None = None
    for ln in lines:
        m = re.match(r"^    ([^:].+):\s*$", ln)
        if m and not ln.strip().startswith("Displays"):
            if current:
                blocks.append(current)
            current = {"block_title": m.group(1).strip(), "raw_lines": []}
            continue
        if current is None:
            continue
        current["raw_lines"].append(ln.strip())
        if "Chipset Model:" in ln:
            current["chipset_model"] = ln.split(":", 1)[1].strip()
        if re.search(r"^\s+Type:\s+", ln):
            current["type"] = ln.split(":", 1)[1].strip()
        if "Vendor:" in ln:
            current["vendor"] = ln.split(":", 1)[1].strip()
        if "Metal Support:" in ln:
            current["metal_support"] = ln.split(":", 1)[1].strip()
        if "Total Number of Cores:" in ln:
            current["gpu_cores"] = ln.split(":", 1)[1].strip()
        if "Bus:" in ln:
            current["bus"] = ln.split(":", 1)[1].strip()
    if current:
        blocks.append(current)

    gpu_blocks = [b for b in blocks if str(b.get("type", "")).upper() == "GPU"]
    selected = gpu_blocks[0] if gpu_blocks else (blocks[0] if blocks else {})
    name = (
        selected.get("chipset_model")
        or selected.get("block_title")
        or "unknown"
    )
    vendor = str(selected.get("vendor") or "")
    metal = str(selected.get("metal_support") or "")
    blob = f"{name} {vendor} {metal}".lower()
    looks_llvmpipe = "llvmpipe" in blob or "lavapipe" in blob or "swiftshader" in blob
    looks_metal_apple = (
        "apple" in blob or "metal" in blob or name.upper().startswith("APPLE")
    )
    return {
        "selected_adapter_name": name,
        "selected_adapter_vendor": vendor or None,
        "selected_adapter_type": selected.get("type"),
        "selected_adapter_metal_support": metal or None,
        "selected_adapter_bus": selected.get("bus"),
        "selected_adapter_gpu_cores": selected.get("gpu_cores"),
        "selected_from": "system_profiler SPDisplaysDataType (first Type: GPU block)",
        "not_llvmpipe": bool(not looks_llvmpipe and looks_metal_apple),
        "llvmpipe_rejection_basis": (
            "name/vendor/Metal strings inspected; llvmpipe/lavapipe/swiftshader absent "
            "and Apple/Metal markers present"
            if (not looks_llvmpipe and looks_metal_apple)
            else "FAILED: software rasterizer markers or missing Apple/Metal markers"
        ),
        "system_profiler_gpu_lines": [
            ln.strip()
            for ln in displays.splitlines()
            if any(
                k in ln
                for k in (
                    "Chipset Model",
                    "Metal Support",
                    "Type:",
                    "Vendor:",
                    "Total Number of Cores",
                    "Bus:",
                )
            )
        ],
    }


displays = subprocess.run(
    ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True
).stdout
swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
memsize = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
adapter = _parse_selected_adapter(displays)

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
            "a_general": np.asarray(fit.a_general, dtype=np.float64),
            "a_specific": np.asarray(fit.a_specific, dtype=np.float64),
            "threshold": np.asarray(fit.threshold, dtype=np.float64),
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

# Call-time interval overlap (Barrier-synced Python threads). This is NOT a
# claim about Metal/wgpu hardware kernel concurrency on the GPU.
span_wall_s = round(mono_t1 - mono_t0, 6)
overlap_timing = {
    "span_wall_s": span_wall_s,  # renamed from mono_overlap_wall_s
    "cpu_mono_interval": [cpu.get("mono_start"), cpu.get("mono_end")],
    "gpu_mono_interval": [gpu.get("mono_start"), gpu.get("mono_end")],
    "simultaneous_interval_s": None,
}
overlap_verdict = {
    "call_time_intervals_overlap": False,
    "basis": "empty_or_failed_arm",
    "interpretation": (
        "call-time interval overlap measures Barrier-synced Python thread wall "
        "intervals only; it does NOT prove hardware kernel concurrency on Metal/wgpu"
    ),
}
if cpu.get("ok") and gpu.get("ok"):
    lo = max(cpu["mono_start"], gpu["mono_start"])
    hi = min(cpu["mono_end"], gpu["mono_end"])
    simultaneous_s = max(0.0, hi - lo)
    overlap_timing["simultaneous_interval_s"] = round(simultaneous_s, 6)
    overlap_verdict = {
        "call_time_intervals_overlap": simultaneous_s > 0.0,
        "simultaneous_interval_s": round(simultaneous_s, 6),
        "basis": "perf_counter intervals of Barrier-synced fit_bifactor_grm calls",
        "interpretation": (
            "call-time interval overlap measures Barrier-synced Python thread wall "
            "intervals only; it does NOT prove hardware kernel concurrency on Metal/wgpu"
        ),
        "hardware_kernel_concurrency_claimed": False,
    }

equality = None
n_compared = 0
n_exceeded = 0
atol = 1e-3
shape_finite_checks: list[dict] = []
if cpu.get("ok") and gpu.get("ok"):
    parts = []
    for key in ("a_general", "a_specific", "threshold"):
        a = np.asarray(cpu[key], dtype=np.float64)
        b = np.asarray(gpu[key], dtype=np.float64)
        shape_ok = a.shape == b.shape
        a_flat = a.ravel()
        b_flat = b.ravel()
        finite_a = int(np.isfinite(a_flat).sum())
        finite_b = int(np.isfinite(b_flat).sum())
        mask = np.isfinite(a_flat) & np.isfinite(b_flat)
        n_slot = int(mask.sum())
        n_compared += n_slot
        diff = np.abs(a_flat[mask] - b_flat[mask]) if n_slot else np.asarray([])
        n_ex = int(np.sum(diff > atol)) if n_slot else 0
        n_exceeded += n_ex
        shape_finite_checks.append(
            {
                "name": key,
                "cpu_shape": list(a.shape),
                "gpu_shape": list(b.shape),
                "shapes_equal": bool(shape_ok),
                "cpu_size": int(a_flat.size),
                "gpu_size": int(b_flat.size),
                "cpu_finite_count": finite_a,
                "gpu_finite_count": finite_b,
                "compared_finite_count": n_slot,
                "all_finite_cpu": bool(finite_a == a_flat.size),
                "all_finite_gpu": bool(finite_b == b_flat.size),
            }
        )
        parts.append(
            {
                "name": key,
                "n": n_slot,
                "max_abs": float(diff.max()) if diff.size else 0.0,
                "n_exceeded_atol": n_ex,
            }
        )
    if cpu.get("loglik") is not None and gpu.get("loglik") is not None:
        n_compared += 1
        ll = abs(float(cpu["loglik"]) - float(gpu["loglik"]))
        n_ex = 1 if ll > atol else 0
        n_exceeded += n_ex
        shape_finite_checks.append(
            {
                "name": "loglik",
                "cpu_shape": [],
                "gpu_shape": [],
                "shapes_equal": True,
                "cpu_size": 1,
                "gpu_size": 1,
                "cpu_finite_count": int(np.isfinite(cpu["loglik"])),
                "gpu_finite_count": int(np.isfinite(gpu["loglik"])),
                "compared_finite_count": 1,
                "all_finite_cpu": bool(np.isfinite(cpu["loglik"])),
                "all_finite_gpu": bool(np.isfinite(gpu["loglik"])),
            }
        )
        parts.append({"name": "loglik", "n": 1, "max_abs": ll, "n_exceeded_atol": n_ex})
    shapes_ok = all(c["shapes_equal"] for c in shape_finite_checks)
    finite_ok = all(
        c["all_finite_cpu"] and c["all_finite_gpu"] for c in shape_finite_checks
    )
    equality = {
        "atol": atol,
        "n_compared": n_compared,
        "n_exceeded": n_exceeded,
        "parity_holds": n_compared > 0 and n_exceeded == 0 and shapes_ok and finite_ok,
        "shapes_equal": shapes_ok,
        "all_compared_finite": finite_ok,
        "shape_finite_checks": shape_finite_checks,
        "parts": parts,
    }

both_converged = bool(cpu.get("converged")) and bool(gpu.get("converged"))
claim_policy = {
    "any_arm_converged_false": not both_converged,
    "research_claim_allowed": False if not both_converged else False,
    "speed_claim_allowed": False,
    "reason": (
        "max_iter=5 smoke fixture; arms report converged=false — "
        "no research inference and no speed/throughput claim from this artifact"
        if not both_converged
        else "smoke fixture only; speed claims still disallowed without dedicated bench protocol"
    ),
    "smoke_only": True,
}

# strip arrays from arms for JSON
for arm in (cpu, gpu):
    for k in ("a_general", "a_specific", "threshold"):
        arm.pop(k, None)

passed = bool(
    equality
    and equality["n_compared"] > 0
    and equality["parity_holds"]
    and adapter.get("not_llvmpipe")
)

report = {
    "artifact": "cpu_gpu_concurrent_live_mba",
    "schema": "cpu_gpu_concurrent_live/v2",
    "schema_notes": (
        "v2: selected adapter (not hardcoded not_llvmpipe); shape/finite/count checks; "
        "separate overlap_verdict; span_wall_s replaces mono_overlap_wall_s; "
        "call-time overlap ≠ HW kernel concurrency; converged=false blocks research/speed claims"
    ),
    "reinforces_commit": "2a8b7f0b0ac2d8313f32d62c9bc76cc433f5c4b0",
    "pass_gate": (
        "n_compared > 0 and parity_holds and shapes_equal and all_compared_finite "
        "and selected adapter not_llvmpipe"
    ),
    "passed": passed,
    "host": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "fast_mlsirm_version": fast_mlsirm.__version__,
    "fast_mlsirm_path": fast_mlsirm.__file__,
    "device": {
        "cpu": "host_cpu",
        "gpu_requested": "gpu",
        "backend_expected": "wgpu→Metal on Darwin arm64",
        **adapter,
    },
    "memory": {"hw_memsize_gb": round(memsize / 1024**3, 2), "vm_swapusage": swap},
    "fixture": {
        **{
            k: COMMON[k]
            for k in (
                "n_cat",
                "n_specific",
                "q_general",
                "q_specific",
                "max_iter",
                "tol",
                "n_starts",
                "seed",
            )
        },
        "n_persons": int(y.shape[0]),
        "n_items": int(y.shape[1]),
        "api": "fast_mlsirm.bifactor_grm.fit_bifactor_grm",
        "bench_path_mirrored": "tests/test_bifactor_gpu.py",
        "note": "TINY identical input; concurrent Barrier(2); NOT A4; no new cargo",
    },
    "timing": {
        "wall_clock_start_utc": wall_clock_start,
        "wall_clock_end_utc": wall_clock_end,
        "overlap_timing": overlap_timing,
        "cpu_wall_s": cpu.get("wall_s"),
        "gpu_wall_s": gpu.get("wall_s"),
    },
    "overlap_verdict": overlap_verdict,
    "arms": {"cpu": cpu, "gpu": gpu},
    "numeric_equality": equality,
    "claim_policy": claim_policy,
    "a4_restarted": False,
    "mac_large_build": False,
}

if equality is None or equality["n_compared"] == 0:
    report["passed"] = False
    report["fail_reason"] = "n_compared=0 or arm failure — cannot pass"
elif not adapter.get("not_llvmpipe"):
    report["passed"] = False
    report["fail_reason"] = "selected adapter failed not_llvmpipe derivation"

OUT.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "passed": report["passed"],
            "n_compared": (equality or {}).get("n_compared"),
            "n_exceeded": (equality or {}).get("n_exceeded"),
            "shapes_equal": (equality or {}).get("shapes_equal"),
            "all_compared_finite": (equality or {}).get("all_compared_finite"),
            "selected_adapter": adapter.get("selected_adapter_name"),
            "not_llvmpipe": adapter.get("not_llvmpipe"),
            "overlap_verdict": overlap_verdict.get("call_time_intervals_overlap"),
            "span_wall_s": overlap_timing.get("span_wall_s"),
            "research_claim_allowed": claim_policy["research_claim_allowed"],
            "speed_claim_allowed": claim_policy["speed_claim_allowed"],
            "out": str(OUT),
            "cpu_ok": cpu.get("ok"),
            "gpu_ok": gpu.get("ok"),
            "cpu_converged": cpu.get("converged"),
            "gpu_converged": gpu.get("converged"),
            "cpu_err": cpu.get("error"),
            "gpu_err": gpu.get("error"),
        },
        indent=2,
    )
)
