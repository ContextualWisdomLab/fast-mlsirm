# Design audit: SOLID · DDD · TDD (2026-09-18)

Branch context: `design-audit-solid-ddd-tdd` worker on checkout tracking
`origin/main` @ `a712995b` (Merge PR #2000 GIL detach). Measurements below
were taken in this workspace unless noted. Remote host `10.6.0.11` was
reachable (`e253504b`, branch `main-sync`) with load averages
**15.60 / 15.71 / 17.42** and local edits under `bifactor_grm.rs` /
`gpu_bifactor.rs`; **no wall-clock benchmarks were run there** (would
contaminate timing and contend with in-flight work).

Priority filter used throughout: unblock #2001 → #2003 → #2004 → #2002;
then numerical/reproducibility risk; aesthetic refactors out of scope.

---

## 1. SRP — `bifactor_grm.rs` change reasons

### Measured size

| Symbol | Start line | ≈ lines | Role |
|---|---:|---:|---|
| file `bifactor_grm.rs` | — | **3359** | module |
| `e_step` | 550 | ~230 | reduced E-step (+ GPU branch) |
| `fit_bifactor_grm` | 999 | ~198 | single-group EM orchestration |
| `e_step_multigroup` | 1780 | ~258 | multigroup E-step (+ GPU) |
| `fit_bifactor_grm_multigroup` | **2302** | **~626** | multigroup EM + validation + EAP |
| `fit_bifactor_grm_fipc` | **2928** | **~432** | FIPC wrapper/reuse |

### Confirmed mixture of reasons (not size-as-defect)

In one module (verified by reading the cited ranges):

1. **Domain estimation** — Bock–Aitkin EM, Gibbons–Hedeker reduction,
   M-step Newton (`m_step_item`, `run_single_start`, `fit_*`).
2. **Input / identification validation** — `validate`,
   `validate_multigroup_cfg`, category-observed checks (~L340,
   multigroup free-item category checks ~L2455).
3. **Execution-device / GPU infrastructure** — direct calls to
   `crate::gpu_bifactor::e_step_reduced_gpu` at **L590** and **L1860**,
   gated on `Device` and `feature = "gpu"`.
4. **Result assembly** — EAP `theta_g_*`, traces, parameter packing into
   `BifactorGrmResult` / `BifactorMultigroupResult` / `BifactorFipcResult`.
5. **Convergence policy** — `checked_em_loglik_change`,
   `refuse_tolerance_on_frozen_start` (#1976), termination reason strings.

### Git history (change-reason churn)

All non-merge commits touching `crates/mlsirm-core/src/bifactor_grm.rs`
in this clone fall in **2026-09** (24 dated entries). Classified by
commit subject (non-merge):

| Reason class | Example SHAs / subjects |
|---|---|
| Numerical / EM policy | `b9b35a0b` false `tolerance_met` on large-q stall; `04b0d9d8` general-only / SE review |
| Quadrature / caller defaults | `25d1aa53` remove GH node-count cap (#1929); `ef79f74e` require `q_*` |
| GPU / device plumbing | `907205c0` GPU E-step; `a8ad30a5` make GPU execute; `ee31b45b` FIPC `device` field |
| API surface growth (FIPC, Oakes SE, multigroup) | `fdeb7b60` stage 1; `0808d59c` stage 2; `2184775a` FIPC; `f7cd5450` Oakes |

**Verdict:** SRP is violated in the *reason-to-change* sense. File size is
a symptom; the defect is that **performance (GPU), numerical policy
(convergence/quadrature), and API growth (FIPC/multigroup)** each require
edits to the same translation unit.

### Recommended boundaries (do not implement as drive-by)

| Boundary | Owns | Does not own |
|---|---|---|
| `bifactor_grm_validate` | loud contracts, category observation, map shape | EM / GPU |
| `bifactor_grm_estep` | CPU reduced E-step + counts/loglik contract | device selection policy |
| `bifactor_grm_em` | multi-start loop, M-step, convergence flags | WGSL / wgpu |
| `gpu_bifactor` (already separate) | device kernels + staging | EM policy |
| `bifactor_grm_fipc` / multigroup facades | composition only | re-implement E-step |

Split only behind characterization tests (see §5 and the companion PR).

---

## 2. DIP — domain → infrastructure

### Direct `gpu_bifactor` coupling

**Fact:** `e_step` / `e_step_multigroup` name `crate::gpu_bifactor` and
construct `ReducedEstepInputs` inline (L571–590, L1841–1860).

**ADR alignment:** ADR-0027 *accepts* this layout: `bifactor_grm` owns the
estimator; `gpu_bifactor` owns WGSL; device is a field on the config; CPU
`f64` remains the reference. ADR-0002 says GPU is a Rust device path, not
a third formula.

**Abstraction cost (not measured on wall-clock this run):** a trait object
or `fn` pointer on the person-sweep would add an indirect call inside a
loop whose body is already tens of thousands of flops per person at
`q=241`. For this hot path, **DIP via dynamic dispatch is not justified
by the ADR and was not performance-validated here**. A static
`enum Device` branch (status quo) or `#[cfg(feature = "gpu")]` monomorphized
helpers is enough. Prefer hoisting fit-invariant staging (#2006 D) over
introducing an E-step trait.

### `parallel.rs` is not a rayon abstraction

**Fact:** `crates/mlsirm-core/src/parallel.rs` (324 lines) implements
**Horn's parallel analysis** (`parallel_analysis`, paran-inspired).
Callers: `factor.rs`, `reliability.rs`.

**Fact:** workspace search of `crates/mlsirm-core` for `rayon`,
`par_iter`, `par_chunks`, `into_par` returned **0 matches** (reconfirmed
this run).

**Verdict:** There is **no empty DIP seam for CPU parallelism**. The
module name collides with everyday “parallel = threads” language and can
mislead #2002 implementers. Rename (e.g. `parallel_analysis.rs`) is a
clarity fix, not an abstraction fill-in.

### GIL note vs #2001

Issue #2001 text says `allow_threads` count is 0. On current `main`
(`a712995b`, PR #2000) the binding layer uses **`Python::detach`**
(PyO3 rename of allow-threads) at bifactor/two-tier/multilevel entry
points (`lib.rs` L1431, L1555, L1680, L1794, L1928, L2042;
`multilevel_bindings.rs` L167/262/362) and
`tests/test_gil_detach_concurrency.py` pins the concurrency contract.
**#2001's “0× allow_threads” claim is stale relative to this checkout**;
remaining #2001 work (if any) should be re-scoped against `detach` coverage
gaps, not a greenfield GIL release.

---

## 3. OCP / LSP — cost of adding a model

### Recent additions (honest file counts)

| Commit | Subject | Files changed |
|---|---|---:|
| `fdeb7b60` | bifactor GRM stage 1 | **13** |
| `f10584de` | two-tier GRM stage 4 | **15** |
| `a3602747` | two-tier Oakes SE (#1994) | **13** |

Typical mandatory touch set for a new Rust-backed fitter:

1. `crates/mlsirm-core/src/<model>.rs` (new)
2. `crates/mlsirm-core/src/lib.rs` (`pub mod`)
3. `crates/fast-mlsirm-py/src/lib.rs` (**~10k-line registry**; imports +
   `#[pyfunction]` + `m.add_function`) — **213** `wrap_pyfunction` /
   `add_function` lines today
4. `python/fast_mlsirm/<model>.py`
5. `python/fast_mlsirm/_legacy_init.py` re-export
6. `tests/unit/<model>_tests.rs` (+ `#[path]` from the Rust module)
7. `crates/mlsirm-core/tests/*` recovery / mirt agreement as applicable
8. `tests/test_<model>.py`, fixtures, changelog fragment

**Verdict:** Open/closed is **weak at the PyO3 registry** (`lib.rs`), not
at the psychometric module naming layer. New models do **not** require
editing sibling estimators (`grm.rs`, `gpcm.rs`, …) for the stage-1/4
pattern — good OCP at the domain-module level — but **do** require editing
the monolithic binding file. LSP was not separately violated in the
sampled bifactor → two-tier reduction test
(`two_tier_reduces_to_bifactor.rs`): substitution is tested where claimed.

---

## 4. DDD — boundaries vs ADRs

### ADRs read and contrasted

| ADR | Claim relevant here | Code match? |
|---|---|---|
| **0001** | Domain-neutral measurement; no hosted runtime | **Yes** — no product HTTP/ORM in core |
| **0002** | Rust owns production numerics; GPU ⊂ Rust device | **Yes** |
| **0027** | `bifactor_grm` + `gpu_bifactor` + bootstrap; CPU reference | **Yes** (direct GPU call is intentional) |
| **0028** | Verb-first public names; **no unsourced defaults** | **Policy**, not module layout; bifactor `q_*` /
  `max_iter` / `tol` already required at Python edge (#1963) |

### Ubiquitous language

- Bifactor public fields consistently use **`theta_g_eap` / `theta_g_sd`**
  (Rust + Python).
- CAT surface still uses **`estimate_ability_*`**
  (`python/fast_mlsirm/cat.py` L326+), while docs/comments often say
  “latent ability” citing Samejima. This is **tolerable IRT diglossia**
  (θ in model math, “ability” in CAT vernacular) but is a real dual
  vocabulary; ADR-0028 rename work should pick one public verb/noun pair
  per surface rather than silently mixing in new APIs.

### Invariant location

Category-observed / ordered-threshold rules are **re-implemented per
model module** (e.g. `bifactor_grm.rs` ~L340, `grm.rs` ~L244,
`gpcm.rs` ~L237, `nominal.rs` ~L250, `mhrm.rs` ~L611,
`two_tier_grm.rs` ~L473) with near-duplicate error strings. That is
**scattered invariant enforcement**, not a single domain type. Risk:
one model’s guard drifts (already a theme in signed-slope / bound work
around #1881).

---

## 5. TDD — what tests fix

### Coverage of cited oversized functions

| Function | Tests that exercise it |
|---|---|
| `fit_bifactor_grm_multigroup` (~626) | `tests/test_bifactor_multigroup.py`, `crates/mlsirm-core/tests/bifactor_multigroup_*.rs`, GPU parity |
| `fit_bifactor_grm_fipc` (~432) | `bifactor_fipc_tests.rs`, `tests/test_poly_fipc.py` |
| `e_step` person loop (~L642) | `estep_gpu_matches_cpu_counts_and_loglik`, #1976 zero-prior test; **until this audit PR, no explicit duplicate-pattern / missing-mask pattern identity test** |
| `simulate_hierarchical_ctar_rasch` (~869) | **Inline** `#[cfg(test)]` in `longitudinal_irt.rs` (simulate+fit round-trips, validation errs) — not absent |
| `fit_marginal_full` / `fit_mhrm` / `fit_2pl` | dedicated `tests/unit/*` suites exist; depth not fully re-audited this run |

### Behavior vs structure

Bifactor suites lean **behavioral**: mirt agreement fixtures, recovery,
bit-reproduce same seed (`test_same_seed_bit_reproduces` in
`tests/test_bifactor_grm.py` / multigroup / two-tier), GPU/CPU envelopes.
That supports refactoring better than snapshot-of-private-structure tests.

### Numerical contracts

Present for bifactor/two-tier (mirt JSON fixtures under
`tests/fixtures/`, `*_mirt_agreement.rs`). Not claimed universal across
every IRT module in this audit.

### Performance contracts

- GIL detach concurrency: `tests/test_gil_detach_concurrency.py` (**exists**
  post-#2000).
- Bootstrap wall-time helpers exist but are measurement/record style
  (`test_bifactor_bootstrap_benchmark.py`), not a CI regression gate for
  per-iteration E-step cost (#2006 F still open).
- **No** crate-level test fails when someone introduces rayon with a
  non-deterministic reduction order (#2002 still needs those gates).

### Determinism

Same-seed bit reproduction is pinned for bifactor/two-tier public fits.
`tests/unit/marginal_distance_tests.rs` pins bit-identical table fills.
#2002’s chunk-count provenance requirement is **not yet** encoded.

---

## 6. Findings → priority

Priority key: **P0** blocks active parallelism track; **P1** correctness /
reproducibility; **P2** structure debt with measured churn; **P3** clarity.

| ID | Priority | Finding | Evidence | Action |
|---|---|---|---|---|
| F1 | **P0** | #2003 lacks an explicit E-step pattern-identity characterization | Hot loop L656–676; prior tests only GPU parity / NaN | **Fixed in companion PR** (duplicate scaling + shared block subvector + missing mask) |
| F2 | **P0/P1** | GPU staging comment denied real deep copies | L567–570 before fix; #2006 type C | **Comment corrected in companion PR** |
| F3 | **P2** | `bifactor_grm.rs` multi-reason churn | §1 git table | Issue: split plan behind tests; no drive-by split |
| F4 | **P2** | PyO3 `lib.rs` registry is the OCP bottleneck | 10 765 lines; every model PR edits it | Issue: continue extracting bindings (pattern already started: `multilevel_bindings.rs` 462, `rotation_bindings.rs` 286, `ata_bindings.rs` 239) |
| F5 | **P3** | `parallel.rs` name ≠ thread parallelism | Horn PA; rayon 0× | Issue: rename module / docs pointer for #2002 readers |
| F6 | **P2** | Category-observed invariants duplicated across models | §4 | Issue: shared validation helper *after* inventory of behavioral differences |
| F7 | **P1** | No CI gate for E-step wall-time / silent GPU fallback provenance | #2006; host too loaded to measure | Leave to #2006 workers; do not invent constants |
| F8 | **Info** | #2001 `allow_threads` wording stale | `detach` present on main | Comment on #2001 / retitle remaining gaps |

---

## 7. What this worker changed

1. Audit report (this file).
2. Characterization tests unblocking #2003 + corrected GPU staging comment
   (companion PR).
3. GitHub issues for F3–F6 (and #2001 clarification) opened from the
   measured rows above.

**Not done (explicitly):** extracting `bifactor_grm` into multiple files;
introducing an E-step trait; running CP3 timing on `10.6.0.11` (load
15+); implementing pattern reduction itself (#2003 implementation).
