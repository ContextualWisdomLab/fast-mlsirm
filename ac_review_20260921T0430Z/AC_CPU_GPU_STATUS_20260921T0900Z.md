# AC CPU/GPU status — what is research, what is library validation (2026-09-21 ~09:00Z)

Direct observation on s1 (seongho@192.168.68.3) and Air (10.6.0.11). All reads were read-only except a tiny synthetic GPU probe in an isolated venv.

## 1. The CPU the user sees is library validation, NOT a research AC fit

| PID (s1) | What it is | cwd |
|---|---|---|
| 453524 `bash run_final.sh` → 453581 `cargo test --workspace` → 1741902 `two_tier_oakes_mirt` | fast-mlsirm **PR #2092** library test suite at `a7e2bf82` | `/data/orca/workspaces/fmls-ac-2092-final-a7e2bf82/src` |
| 1521532 `run_py2b.sh` → 1543930 `pytest` | the same PR's Python suite, isolated clone/venv | `…/fmls-ac-2092-final-a7e2bf82/src-py2` |

- They use no research data, write no research outputs, and fit nothing that is adopted.
- The 90-minute part was `bifactor_oakes_calibration` (100 simulated replicates, single-threaded). It **passed**: 2 ok, 1 ignored, 5398.28 s.
- **No research AC process runs anywhere now.** On s1 the W run pid 3788557 has exited; Air has no AC or late-life process.

## 2. Research AC (W) actual state

| Item | Fact |
|---|---|
| Input | `late-life-w-ac-mg-s1-20260921/local/avoidance_coping_items.csv` (13 items), N=1020, age-group masks |
| Fit done | `fit_bifactor_grm_multigroup`, concurrent MG, `anchor_mask=None`, q=121/121, `max_iter=2000`, `tol=1e-6`, `n_starts=1`, **`device="cpu"` requested**, **no slope prior (MML)**. Log: `[done] W mg n_iter=121` |
| Preserved fit | `local/library_score_corr_checkpoints_W/W_ac_mg_q121_seed11400714819323198485.pkl`, sha256 `d74eb91a…1a16c81411` (21088 B), re-verified now |
| Failure | **scoring only, after the fit**: `check_bifactor_expected_total_score_monotonicity` → `fit.a_general must be a non-empty 1-D array` (the MG fit is `(3,13)`) |
| Loaded library | `/data/orca/workspaces/fmls-perf-diag-venv`. fast_mlsirm reports "0.11.4", but it was **installed from the local dev wheel** `file:///…/fmls-g4w-fipc-s1-wheels/32807ed0/…manylinux_2_35….whl`, `_core` sha16 `a0453b7a`. It is NOT the PyPI release. |
| Refit needed? | **No.** The fit is valid and preserved; only expected-raw scoring is missing. |
| Next step | Score the preserved pkl with the MG-aware predictor (fast-mlsirm **#2091**, `predict_bifactor_expected_total_score`). The #2091 session already showed max abs diff 0.0 vs the grid workaround on this pkl. It is **blocked on an immutable release carrying #2091**, not on compute. The consumer migration patch is staged unapplied: `local/branch_verify/proposed_expected_raw_migration.patch`, owned by the #2091 session / CO. |
| MG sensitivity vs adopted slope prior | This W run is the **MG (age-group concurrent) MML** path. The **AC lognormal slope prior** (PR #2092, MAP) is a separate, **not adopted** option: F069 is open, and no research fit with a prior has been run or requested. Do not conflate the two. |

## 3. GPU facts (measured, not assumed)

- **s1:** the NVIDIA GTX 1050 (GP107) is present, but the loaded kernel module is 580.159.03 while the installed userspace driver is 580.178.04, so `nvidia-smi` reports "Driver/library version mismatch".
  - Vulkan exposes only `llvmpipe` (`PHYSICAL_DEVICE_TYPE_CPU`).
  - Probe in the isolated venv (synthetic 300×6, q=7): `device="gpu"` ran the f32 WGSL path **with no fallback warning**, loglik `-1971.5175023` vs CPU `-1971.5175400`, **0.93 s vs 0.04 s**. So "gpu" on s1 is a CPU software renderer.
  - Unblocking the real GPU needs a host-admin driver reload or reboot. That is not this owner's action.
- **Library finding (follow-up, not fixed here):** `GpuContext::init` (`crates/mlsirm-core/src/gpu.rs`) takes the default adapter without checking `DeviceType`. It therefore silently accepts a CPU software adapter as "gpu", with no warning.
- **Air:** Apple M5, Metal 4. It is GPU-capable for wgpu, but no AC run is needed there: the next research step is scoring, which is light CPU work, and no refit is justified.

## 4. Library validation (PR #2092) — separate table

| Stage | Result |
|---|---|
| `cargo test --workspace` (native, s1) | running; ≥1157 ok, 0 failed so far; Oakes calibration 2 ok / 1 ignored (5398 s) |
| py crate `cargo test` (s1, target-py) | EXIT 0, 9 passed |
| isolated Python: pip --require-hashes | EXIT 0 |
| isolated Python: editable install onward | running (`run_py2b.sh`) |
| hosted CI | 231d owns it |
