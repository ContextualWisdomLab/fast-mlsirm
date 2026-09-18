# E-step nest audit ledger (#2004)

Inventory of ≥3-deep nests that index `(person, item)` via `p * n_items + i`
or `is_obs(p, i)` inside `crates/mlsirm-core/src` (issue #2004 / follow-on to
#2003). Nest start lines are the issue inventory checked against
`origin/main` @ `a712995b` (2026-09-18); drifted starts are noted.

## Defect classes

| Code | Meaning |
|------|---------|
| **A** | Loop-invariant person/item lookup inside quadrature nest (`is_obs` / `y[…]` under `g,h`) |
| **B** | Innermost walks items (stream crossing) instead of contiguous node stream |
| **C** | Table layout `[node][cat]` wastes cache / blocks vectorization on `h` |
| **D** | Large intermediate (`block_acc`) then immediate axis reduction |
| **P** | Response-pattern collapse applicable (Bock & Aitkin, 1981, pp. 445, 448; bifactor block CI: Gibbons & Hedeker, 1992, pp. 423, 425; Gibbons et al., 2007, pp. 5, 7–8) — owned by #2003 / PR #2005 |

## Method

1. Profile a **representative bifactor GRM** CPU E-step (CP3-shaped design,
   scaled quadrature) with section timers before editing.
2. Fix only measured hotspots; one concern per PR; measure before/after.
3. Cold nests stay in this ledger as `ignore (measured cold)` / `deferred`.
4. No unsourced numeric defaults (ADR-0028). Algorithmic reductions cite papers;
   pure loop-invariant hoists cite this ledger + measured timings.

## Profile workload (bifactor)

| Knob | Value | Basis |
|------|-------|-------|
| `n_persons` | 240 | scaled CP3-like |
| `n_items` / blocks | 13 / 3×4 + 1 general-only | CP3 layout |
| `n_cat` | 4 | CP3 |
| `q_general` = `q_specific` | 41 | study-scale proxy (not a default; caller-owned in API) |
| device | CPU | hot-path owner under audit |
| repeats | 5 timed E-steps after 1 warmup | wall clock |

Full CP3 (`N=1020`, `q=241`) is hours on one core; relative nest ranking is
stable in `q` because the dominant term is `O(N · S · q_g · q_s · m_block)`.

## Bifactor GRM ledger

| Nest start | Role | Depth | A | B | C | D | P | Action |
|------------|------|------|---|---|---|---|---|--------|
| `bifactor_grm.rs:642` | CPU `e_step` person sweep | 4 | Y | Y | Y | Y | Y (#2003) | Profiled; **counts nest hotter than block_acc**; this PR skips counts on loglik-only |
| `bifactor_grm.rs:1087` | final EAP `block_acc` | 4 | Y | Y | Y | Y | Y (#2003) | deferred (needs counts/EAP) |
| `bifactor_grm.rs:1321` | brute full-grid oracle | 3 | Y | n/a | n/a | n/a | n/a | ignore (oracle cap; not EM hot path) |
| `bifactor_grm.rs:1949` | multigroup `e_step` `block_acc` | 4 | Y | Y | Y | Y | Y (#2003) | deferred |
| `bifactor_grm.rs:2439` | free-item category validation | 3 | n/a | n/a | n/a | n/a | n/a | ignore (once per fit) |
| `bifactor_grm.rs:2584` | multigroup final EAP `block_acc` | 4 | Y | Y | Y | Y | Y (#2003) | deferred |
| `bifactor_grm.rs:3264` | FIPC final EAP `block_acc` | 4 | Y | Y | Y | Y | Y (#2003) | deferred |

### Measured section share (baseline WITH expected counts)

Host run 2026-09-18, `estep_nest_profile_2004` binary, workload above,
`accumulate_counts = true` (pre-fix path):

| Section | ms | Share of timed sections | Notes |
|---------|----|-------------------------|-------|
| `block_acc` + per-block `log_sum_exp` | 153.0 | 33.0% | |
| general-only accumulation | 0.06 | ~0% | |
| posterior / expected counts | 311.1 | 67.0% | **hottest** |
| wall (one E-step via marginal API) | 469.9 | — | includes table setup outside timers |

### Defect A hoist — measured, NOT applied

Isolated `block_acc` microbench (`q=41`, 4 items, fully observed / `observed=None`
style): hoist was **0.20×–0.50×** vs naive (regression). With a 1/4 missing
mask the result was noisy (~0.9×–2.0×). Decision: leave lookups in the nest;
do not ship defect-A hoist for this path.

### Before / after — skip expected counts on loglik-only

| Metric | Before (`accumulate_counts=true`) | After (`false` in `bifactor_grm_marginal_loglik`) |
|--------|-----------------------------------|--------------------------------------------------|
| wall ms (same fixture) | 469.9 | 33.2 (~14.1×) |
| posterior/count ms | 311.1 | 0.17 (loglik combine only) |
| block_acc ms | 153.0 | 30.6 |
| block_acc share of timed | 33% | 99.3% |
| observed-data loglik | −4503.474215 | −4503.474215 (identical) |

Fit / Oakes callers keep `accumulate_counts = true` (counts required for M-step).

**B/C/D** left untouched in this PR so regressions stay attributable.
**P** remains #2003 (PR #2005).

## Other files (issue inventory) — deferred

Not on the bifactor representative profile. Static A/B/C/D filled; **no code
change** until a workload-specific profile says otherwise.

| Nest start | Notes (static) | Profile |
|------------|----------------|---------|
| `cdm.rs:362,688,1776,1884,2127,2231,2688,2788,3204,3299` | EM / person loops; separate CDM product path | deferred |
| `crm.rs:241,332` | CRM EM | deferred |
| `dif.rs:2101,2133` | DIF item loops | deferred |
| `facets.rs:158,602` | facets | deferred |
| `fitstats.rs:671,1033,1287,1649,3025` | fit statistics | deferred |
| `lltm.rs:464` | LLTM EM | deferred |
| `marginal.rs:485,1459,1568,1728,1818` | MLSIRM marginal E-step | deferred (own benches under `docs/benchmarks/`) |
| `mixture.rs:341,354,382,613` | mixture | deferred |
| `mmle.rs:211` | MMLE | deferred |
| `oakes.rs:160` | Oakes SE | deferred |
| `rt.rs:204` | response-time | deferred |
| `two_tier_grm.rs:838,1517,1836` | line drift vs issue list; streaming E-step | deferred (#2003 follow-up note) |
| `twopl.rs:850,1063` | 2PL | deferred |

## Regression guard (follow-up)

Prefer a `clippy` / custom lint for `is_obs` / `y[p * n_items + i]` lexically
inside bodies that also bind `for g`/`for h` quadrature counters. If not
feasible in CI, record an ADR successor to this ledger rather than weakening
gates.

## References

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
  item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
  443–459. (pp. 445, 448 — pattern-frequency EM)
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
  analysis. *Psychometrika, 57*(3), 423–436. (pp. 423, 425)
- Gibbons et al. (2007). Full-information item bifactor analysis of graded
  response data. *Applied Psychological Measurement, 31*(1), 4–19. (pp. 5, 7–8)
- ADR-0028 public API naming and defaults policy (no unsourced constants)
