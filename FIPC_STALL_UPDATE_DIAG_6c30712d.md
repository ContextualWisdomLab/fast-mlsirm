# FIPC stall update-path diag @6c30712d

**Evidence:** `FIPC_STALL_UPDATE_DIAG_6c30712d.json` (also `/tmp/` and `.../fmls-g4w-fipc-s1-verify-run/logs/` on s1)
**Process:** PID 3552626 used **canonical** `.venv`; loaded `_core` sha16=`6a2df43e`. Exited ~10:16+09 after ~14m. `fmls-perf-diag-venv` was empty (no module) — isolation incomplete for this run.

## Fit summary
- `termination_reason=max_iter_reached`, `n_iter=100`, `n_accepted_prior_steps=100`, `n_rollback_full=0`
- `final_ll == final_fixed_ll` (−1390.24) for entire trace (`ll_equals_fixed_entire_trace=true`) — remapped coords in fixed E-step

## Stall mechanism
- **Mean frozen rows 0→95** at `[0.248, -0.024]`; first move at 96 → `[0.320, -0.040]`
- During freeze: `cov_delta_max≈0.102` (scale-only path accepts); fixed_ll improves −1404.69 → −1398.74
- Early targets already want mean ≈`[0.327, -0.037]` with `mean_tgt_gap≈0.081` while mean stays put

## Candidate probes (Python GH remapped LL @ freeze proxy)
| candidate | ll | vs current (−1398.807) |
|---|---|---|
| joint_target | −1397.916 | **+0.891 better** |
| mean_only_full | −1397.953 | **+0.854 better** |
| mean_only_0.1 | −1398.689 | +0.118 better |
| mean_only_0.01 | −1398.795 | +0.012 better |

FD diagnostic (not a writer change): directional FD toward tgt−mean ≈ **+1.21**; coord FD ≈`[+14.67, −3.30]`.

## Reject-path interpretation (from diag)
- Joint full step rejected early while mean frozen and `n_accepted` grows via scale-only
- Mean-only recovery rejected until ~row 96 despite probe LL improvements
- PD not the blocker
- Accept demands strict remapped improvement vs fixed_ll; with `fixed_ll≡ll` this is same-measure — probes show improvement exists, so Rust accept/backtrack path is failing to take mean steps that the Python probe says are better

## Consumer non_unit audit
- Final mean `[0.324,−0.041]`, sd `[0.915, 0.959]`
- Gate definition in diag still `max(|mean|)>0.1 AND max(sd)>1.01` → **current_gate_pass=false** while `sd_lo` (fixture scale 0.8) is invisible to one-sided max(sd)>1.01
- Confirm consumer REQUIRED_GATES uses two-sided sd (max>1.01 **or** min<0.99)

## Next (Air / numerical) — no fixture-as-optimizer
1. Fix early mean-step rejection so candidates that improve remapped Class-A LL are accepted (joint or mean-only), without fixture-target losses.
2. Preserve FIPC target + LL guard; stall-exit ≠ accept.
3. After patch: isolated verify via `fmls-perf-diag-venv` (install there), then canonical s1 consumer @ tip.
