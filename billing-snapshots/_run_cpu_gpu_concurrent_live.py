"""Live same-input concurrent CPU+GPU measure (tiny fixture; no cargo).

Mirrors tests/test_bifactor_gpu.py fit_bifactor_grm(device=cpu|gpu) contract
with threading.Barrier start sync. n_compared counts finite parameter slots.

Evidence contract (root reviews of 2a8b7f0b / 686dcab5):
- system_profiler Type:GPU is *host_gpu_inventory only* — not library/wgpu selection.
- Library/wgpu adapter+backend remain unverified until a fast-mlsirm diagnostic API
  or adapter logs exist (0.11.4 wheel exposes none).
- Pass gate = numeric smoke parity (shape/finite/count); NOT inventory not_llvmpipe.
- Separate overlap_verdict; span_wall_s; call-time overlap ≠ HW kernel concurrency.
- converged=false ⇒ research/speed claims disallowed (smoke only).
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fast_mlsirm.bifactor_grm import fit_bifactor_grm
import fast_mlsirm

OUT = Path(__file__).with_name("cpu_gpu_concurrent_live_mba_20260920T1610.json")

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


def _parse_host_gpu_inventory(displays: str) -> dict:
    """Host display/GPU inventory from SPDisplaysDataType (not wgpu selection)."""
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
    name = selected.get("chipset_model") or selected.get("block_title") or "unknown"
    vendor = str(selected.get("vendor") or "")
    metal = str(selected.get("metal_support") or "")
    blob = f"{name} {vendor} {metal}".lower()
    looks_llvmpipe = "llvmpipe" in blob or "lavapipe" in blob or "swiftshader" in blob
    return {
        "inventory_adapter_name": name,
        "inventory_adapter_vendor": vendor or None,
        "inventory_adapter_type": selected.get("type"),
        "inventory_adapter_metal_support": metal or None,
        "inventory_adapter_bus": selected.get("bus"),
        "inventory_adapter_gpu_cores": selected.get("gpu_cores"),
        "inventory_from": "system_profiler SPDisplaysDataType (first Type: GPU block)",
        "inventory_looks_like_llvmpipe": bool(looks_llvmpipe),
        "inventory_note": (
            "Host GPU inventory only. Does NOT identify the adapter actually "
            "selected by fast-mlsirm/wgpu."
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


def _probe_library_wgpu_selection() -> dict:
    """Attempt real adapter/backend observability from the installed library.

    0.11.4 exposes no adapter diagnostic API on ``fast_mlsirm._core``. Absence
    is recorded as unverified — never inferred from host inventory.
    """
    from fast_mlsirm import _core

    attrs = [a for a in dir(_core) if any(k in a.lower() for k in ("gpu", "wgpu", "adapt", "device"))]
    return {
        "status": "unverified",
        "reason": (
            "installed fast-mlsirm has no public adapter/backend diagnostic API; "
            "wgpu Python package not required/installed for this wheel path"
        ),
        "core_gpuish_attrs": attrs,
        "fast_mlsirm_version": fast_mlsirm.__version__,
        "needs": (
            "generic observability: adapter name + backend from wgpu AdapterInfo "
            "or library diagnostic logs at GPU context init"
        ),
    }


displays = subprocess.run(
    ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True
).stdout
swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
memsize = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
host_gpu_inventory = _parse_host_gpu_inventory(displays)
library_wgpu_selection = _probe_library_wgpu_selection()

box: dict = {}
arm_warnings: dict[str, list[str]] = {"cpu": [], "gpu": []}
barrier = threading.Barrier(2)
wall_clock_start = datetime.now(timezone.utc).isoformat()
mono_t0 = time.perf_counter()


def run(device: str) -> None:
    barrier.wait()
    t0 = time.perf_counter()
    wall0 = datetime.now(timezone.utc).isoformat()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_bifactor_grm(**COMMON, device=device)
            arm_warnings[device] = [str(w.message) for w in caught]
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
            "warnings": arm_warnings[device],
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
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
            "warnings": arm_warnings.get(device, []),
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

span_wall_s = round(mono_t1 - mono_t0, 6)
overlap_timing = {
    "span_wall_s": span_wall_s,
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
    "hardware_kernel_concurrency_claimed": False,
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

# Weak signal only: GPU arm warning about missing adapter (still not AdapterInfo).
gpu_warn_text = " | ".join(gpu.get("warnings") or [])
cpu_fallback_warned = "no usable GPU adapter" in gpu_warn_text or "fall back" in gpu_warn_text.lower()
library_wgpu_selection["gpu_arm_fallback_warning_observed"] = bool(cpu_fallback_warned)
library_wgpu_selection["gpu_arm_warnings"] = list(gpu.get("warnings") or [])
library_wgpu_selection["note"] = (
    "Absence of CPU-fallback warning is NOT proof of Metal adapter identity; "
    "adapter name/backend remain unverified without diagnostic API."
)

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
        a_flat, b_flat = a.ravel(), b.ravel()
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
                "shapes_equal": bool(a.shape == b.shape),
                "cpu_size": int(a_flat.size),
                "gpu_size": int(b_flat.size),
                "cpu_finite_count": int(np.isfinite(a_flat).sum()),
                "gpu_finite_count": int(np.isfinite(b_flat).sum()),
                "compared_finite_count": n_slot,
                "all_finite_cpu": bool(np.isfinite(a_flat).all()),
                "all_finite_gpu": bool(np.isfinite(b_flat).all()),
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
    finite_ok = all(c["all_finite_cpu"] and c["all_finite_gpu"] for c in shape_finite_checks)
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
    "research_claim_allowed": False,
    "speed_claim_allowed": False,
    "reason": (
        "max_iter=5 smoke; converged=false — no research/speed claim; "
        "do not treat as research-complete compute path"
    ),
    "smoke_only": True,
}

for arm in (cpu, gpu):
    for k in ("a_general", "a_specific", "threshold"):
        arm.pop(k, None)

passed = bool(equality and equality["n_compared"] > 0 and equality["parity_holds"])

report = {
    "artifact": "cpu_gpu_concurrent_live_mba",
    "schema": "cpu_gpu_concurrent_live/v3",
    "schema_notes": (
        "v3: host_gpu_inventory renamed (was selected_adapter); library_wgpu_selection "
        "explicitly unverified; pass_gate is numeric parity only — inventory must not "
        "satisfy not_llvmpipe/adapter identity"
    ),
    "reinforces_commit": "686dcab53543baf307d798908b9f05d4b5c8b036",
    "pass_gate": "n_compared > 0 and parity_holds and shapes_equal and all_compared_finite",
    "passed": passed,
    "host": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "fast_mlsirm_version": fast_mlsirm.__version__,
    "fast_mlsirm_path": fast_mlsirm.__file__,
    "device": {
        "cpu": "host_cpu",
        "gpu_requested": "gpu",
        "backend_expected": "wgpu→Metal on Darwin arm64 (expected, not verified)",
        "host_gpu_inventory": host_gpu_inventory,
        "library_wgpu_selection": library_wgpu_selection,
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
        "note": "TINY identical input; concurrent Barrier(2); NOT A4; no new cargo; smoke only",
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

OUT.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "passed": report["passed"],
            "schema": report["schema"],
            "n_compared": (equality or {}).get("n_compared"),
            "inventory_name": host_gpu_inventory.get("inventory_adapter_name"),
            "library_wgpu_status": library_wgpu_selection.get("status"),
            "gpu_fallback_warned": library_wgpu_selection.get(
                "gpu_arm_fallback_warning_observed"
            ),
            "research_claim_allowed": False,
            "out": str(OUT),
        },
        indent=2,
    )
)
