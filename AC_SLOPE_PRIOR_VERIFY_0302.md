# Independent verify — AC SlopePrior::Lognormal on |a| (updated 03:04Z)

## Status: NOT ACCEPTED (HOLD)

Root 03:04: `|a|` lognormal is a **signed-support extension**, not adopt-by-Chalmers-direction. Need primary-source support range, Jacobian, 0-boundary, and fixed-anchor exclusion before acceptance.

## Source read
MBA WT `fmls-ac-slope-prior` — `SlopePrior::{None,Lognormal{mu,sd}}`, `add_lnorm_abs_slope_prior` on `|a|` with sign chain-rule; cites Chalmers (2012) mirt `priorType=lnorm`.

## Open checks (blocking)
1. **Support:** mirt/Chalmers lnorm is on positive discriminations; mapping to unconstrained signed `a` via `|a|` needs explicit Jacobian/`d log p / d a` justification vs placing lnorm on positive reparam only.
2. **0-boundary:** `u=max(|a|,1e-12)` — document effect on Newton near zero; confirm not a silent clamp of the parameter.
3. **Fixed anchors:** prior must not apply to FIPC/fixed-item slopes; confirm call sites skip anchors.
4. **FD:** analytic grad of `add_lnorm_abs_slope_prior` vs finite difference.
5. **Surfaces:** Python `fit_bifactor_grm` + MG path same knobs; `None` ≡ MML identity test; no research defaults.

## Air mem (for scorer, not AC gate)
See lead 03:04 pressure measurement (available/memory_pressure/RSS/swap) — free pages alone not used.
