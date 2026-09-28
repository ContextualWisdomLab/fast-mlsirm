# HETERO CPU+GPU concurrent × Valkey distributed — call paths & unmet contracts

**Status:** READ-ONLY map (2026-09-21). No writer edits, builds, merges, or CI claims.
**Scope:** Actual call paths and **unmet** contracts for two-tier / FIPC-adjacent surfaces under hetero CPU+GPU and Valkey distribute.
**Sources:** `fmls-2001-l4-fit-score-wire` remote stack (Valkey + worker families), lead `billing-snapshots/*`, `mlsirm-core` `Device` / GPU path, poly FIPC / two-tier APIs.

---

## 1. Axes (do not conflate)

| Axis | What it is | What it is not |
|------|------------|----------------|
| **Remote family** | Whole-call unit under `RemoteJobFamily` + `remote_worker.execute_*` | Internal EM/M-step sharding of one FIPC/two-tier fit |
| **Valkey distribute** | `SubprocessExecutor` + `ValkeyStreamsOutcomeStore` / commit ledger over Streams | Research convergence or manuscript cells |
| **`ExecutionFloatPath`** | Cohort manifest tag `f64` \| `f32` | Automatic GPU selection |
| **`FitConfig.rust_device`** | MLSIRM binary `fit` / objective: `{cpu,gpu,auto}` → `Device` / wgpu | Poly FIPC, two-tier GRM, WLE, bifactor Oakes |
| **CPU+GPU concurrent** | Same wall-clock cohort with workers on CPU and GPU (or dual-device hosts) with declared float/device identity | “GPU available somewhere” or silent `auto` fallback |

---

## 2. Actual call paths (as implemented)

### 2.1 Local MLSIRM GPU (in-process)

```
FitConfig(backend=rust|auto, rust_device={cpu,gpu,auto})
  → fit / objective
  → mlsirm_core::neg_loglik_and_grad_device(Device)
       Cpu  → f64 scalar
       Gpu|Auto → wgpu f32 kernels when adapter present; else CPU fallback
```

**Evidence bound:** CI/gpu-smoke style paths; **not** hetero Valkey 3-host evidence.

### 2.2 Valkey / subprocess remote (hetero)

```
Driver: SubprocessExecutor.run_batch(envelopes, worker_manifest, payload)
  → SSH/local worker entry → remote_worker.main
  → RemoteJobEnvelope + payload_sha256 check
  → execute_envelope(family) → library function
  → OutcomeCommitLedger / ValkeyStreamsOutcomeStore (at-most-once commit by envelope fingerprint)
```

**Legal families today** (`RemoteJobFamily`):

| Family | Worker entry | Library function | Device / GPU knobs |
|--------|--------------|------------------|--------------------|
| `mc_replicate` | `execute_mc_replicate` | `simulate` | None |
| `fit_restart` | `execute_fit_restart` | `fit` + `FitConfig(**payload["config"])` | **Only if** payload config includes `rust_device`; not required by contract |
| `em_m_step` | `execute_em_m_step` | `fit` max_iter=1 | Same as fit |
| `scoring_person` | `execute_scoring_person` | `score_wle` | None (CPU) |
| `se_derivatives` | `execute_se_derivatives` | `bifactor_oakes_se` | None |
| `regression_contrasts` | `execute_regression_contrasts` | OLS/HC + contrast | None |
| `fipc` | `execute_fipc` | `fit_poly_fipc` | **None** — Rust poly MML-EM, no `Device` |
| `fipc_group_person_score` | `execute_fipc_group_person_score` | `score_poly_fipc_group_persons` / payload executor | **None** — 1-D poly GRM + GH EAP |
| `two_tier` | `execute_two_tier` | `fit_two_tier_grm` | **None** — no `rust_device` in API |

`RemoteRunManifest.float_path` is a **cohort identity** field (`f64`/`f32`). It is **not** wired in `remote_worker` to set `rust_device` or to refuse GPU/CPU mix.

### 2.3 FIPC / two-tier adjacent (local APIs vs remote)

| Surface | Local API | Remote family | Valkey 3-host evidence |
|---------|-----------|---------------|------------------------|
| 1-D poly FIPC calibrate | `fit_poly_fipc` | `fipc` | **ABSENT** |
| 1-D poly FIPC group EAP + expected-raw + anchor ref moments | `fipc_group_score.*` (#2079) | `fipc_group_person_score` | **ABSENT** |
| Two-tier GRM fit | `fit_two_tier_grm` | `two_tier` | **ABSENT** |
| Two-tier `E[T\|θ_f]` nuisance curve (#2077) | `expected_total_score_two_tier_given_primary` | **No family** | **ABSENT** |
| G+4+W / research two-tier FIPC | OpenCode / `fmls-g4w-fipc` (out of HETERO write lane) | Not this map’s implementer | N/A |

---

## 3. What hetero evidence already covers (and what it does **not**)

| Artifact (lead `billing-snapshots/`) | Claim level | Families | Valkey product | CPU+GPU concurrent |
|--------------------------------------|-------------|----------|----------------|--------------------|
| `valkey_fail_retry_*.json` | Transport deliberate fail→XAUTOCLAIM | Protocol probe (not library) | Disposable broker | No |
| `valkey_library_3host_mc_replicate_*.json` | Generic library path | `mc_replicate` only | Was redis:7-alpine in amend path; later Valkey 8.1.10 for fit/score | No |
| `fit_score_*_unified_valkey_*.json` | Generic / later **TRANSPORT smoke** | `fit_restart` + `scoring_person` (often `max_iter=1`) | Valkey 8.1.10 `@192.168.68.3:16382` | No |
| `fit_score_single_vs_3host_parity_wheel_*.json` | TRANSPORT/DISPATCH parity + same-result reclaim/dedup smoke | fit + score | Valkey 8.1.10 | No |
| Reclaim note | **Not** dead-consumer XAUTOCLAIM failure-recovery complete | — | — | — |

**No** billing snapshot claims: `two_tier`, `fipc`, `fipc_group_person_score`, `#2077` curve, or **CPU+GPU concurrent** cohorts over Valkey.

---

## 4. Unmet contracts (actionable gaps)

### U1 — Valkey distribute for FIPC / two-tier families
- **Required:** 3-host (or ≥2-host) `SubprocessExecutor` + Valkey Streams commit for `fipc`, `fipc_group_person_score`, and/or `two_tier` with real library import + per-host core SHA + numeric equality definition.
- **Present:** Worker handlers exist; Valkey path proven only for `mc_replicate` / fit+score smoke.
- **Status:** UNMET.

### U2 — Remote family for #2077 expected-raw curve
- **Required:** If research consume needs remote `E[T|θ_f]`, admit a whole-call family (or explicit non-remote contract).
- **Present:** Local API only; no `RemoteJobFamily` member.
- **Status:** UNMET (or explicitly out-of-scope — must be declared).

### U3 — CPU+GPU concurrent hetero
- **Required:** Cohort policy that binds `float_path` **and** effective device (`requested_device` / `effective_device` already appear on provenance helpers) so CPU and GPU workers cannot silently mix incompatible numerics; evidence of concurrent CPU+GPU runs with equality or documented tolerance.
- **Present:** GPU only on MLSIRM `fit` via `rust_device`; poly/two-tier have no device; manifest `float_path` unused by worker device selection; no concurrent CPU+GPU Valkey evidence.
- **Status:** UNMET.

### U4 — Device plumbing for remote `fit_restart` under GPU
- **Required:** If GPU hosts join fit cohorts, payload/`FitConfig.rust_device` + manifest must match, and fallbacks (`auto`→CPU) must be recorded in outcome provenance for equality gates.
- **Present:** Optional config passthrough only; no hetero GPU fit evidence.
- **Status:** UNMET for GPU; CPU fit smoke only.

### U5 — Library-path failure-recovery (XAUTOCLAIM)
- **Required:** Forced consumer kill → XAUTOCLAIM → re-run → dedup on a **library** family (fit/score/FIPC/two-tier), separate from same-result reclaim smoke.
- **Present:** Protocol `valkey_fail_retry_*`; fit/score reclaim labeled transport/dedup smoke only.
- **Status:** UNMET (separate owner/task; not claimed complete).

### U6 — Research G+4+W two-tier FIPC vs library 1-D poly FIPC
- **Required:** No dual-write; share arg contracts only. G+4+W FIPC ≠ `#2079` 1-D poly group person-score.
- **Present:** Boundary documented in #2079 / orchestration; G4W owned elsewhere (`fmls-g4w-fipc` / tip builds out of HETERO write lane).
- **Status:** Contract boundary MET as documentation; research consume still SEPARATE.

---

## 5. Minimal “done” definitions (for a future implement/evidence lane)

1. **Valkey FIPC or two-tier:** PASS artifact with `family` ∈ {`fipc`,`fipc_group_person_score`,`two_tier`}, Valkey product ID, per-host library/core SHA, equality rule, label ≠ research-complete.
2. **CPU+GPU concurrent:** PASS artifact listing host×`effective_device`, cohort `float_path`, and either bit-identity or documented f32/f64 bound — for a family that actually uses `Device`.
3. **Failure-recovery:** PASS artifact with kill→XAUTOCLAIM→library re-run→dedup on Valkey, distinct from parity reclaim smoke.
4. **#2077 remote (optional):** New family **or** written “local-only” waiver signed by research lead.

---

## 6. Explicit non-claims

- This file does **not** authorize builds, merges, admin-merge, shared Valkey restarts, or A4 restart.
- Tip FIPC / s1 verify SHAs (e.g. G4W trees) are **out of HETERO write lane** — observation only.
- `#2079` CI watch **stopped** per root reassign; no merge claim from this document.
- Prior Valkey fit/score PASS remains **transport / dispatch / dedup smoke**, not psychometric distributed compute complete, not CPU+GPU concurrent.

---

## 7. Pointers (read-only)

| Path | Role |
|------|------|
| `python/fast_mlsirm/remote_exec.py` | Families, manifest/`float_path`, SubprocessExecutor, ledgers, Valkey store |
| `python/fast_mlsirm/remote_worker.py` | `execute_*` dispatch |
| `python/fast_mlsirm/fipc_group_score.py` | 1-D poly FIPC group person-score + anchor ref moments |
| `python/fast_mlsirm/polytomous.py` (`fit_poly_fipc`) | 1-D FIPC calibrate (no GPU device) |
| `python/fast_mlsirm/two_tier_grm.py` (`fit_two_tier_grm`) | Two-tier fit (no GPU device) |
| `crates/mlsirm-core` `Device` / `neg_loglik_and_grad_device` | MLSIRM GPU path only |
| `billing-snapshots/fit_score_single_vs_3host_parity_wheel_20260920.json` | Transport parity + reclaim smoke label |
| `billing-snapshots/valkey_fail_retry_20260920.json` | Protocol XAUTOCLAIM (not library FIPC/two-tier) |
