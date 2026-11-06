# Rust accept-path capture @3fc6160a

## Baseline tip (pre-repair)

- crate tip SHA: `3fc6160a0f5f2be8df6349e0a1070b2c55958897`
- inputs: tiny two-tier FIPC fixture (n_persons=60, Qp=Qs=7, max_iter=8)
- prior init: mean=0, cov=I
- PD: candidate cov PD true on recorded recovery steps

### Selected tip decision (real Rust objective)

`iter=1;branch=mean_accept;alpha=1.000e-1;ll=-7.3751378502e2;fixed_ll=-7.3751378502e2;mean_ll=-7.3685270321e2;scale_first=true;mean_pd=true;mean=[-0.0309022991552216, -0.09455485007522234]`

- before LL (`ll` / `fixed_ll`): `-737.51378502` (equal; remapped Class-A measure)
- candidate mean LL: `-736.85270321` (improves under existing LL guard)
- branch: scale ran first (`scale_first=true`), then mean accepted only at trust-region `alpha=0.1`
- tip never tried `alpha=1.0` for mean even though the remapped objective improved

### Tip decision summary

- iter 0: `joint_full_accept`
- iters 1–7: `mean_accept` at `alpha=1.000e-1` with `scale_first=true`
- mean crawl: `[-0.026,-0.077]` → `[-0.054,-0.184]` over 8 iters

### Root cause (Rust path, not Python proxy)

1. Pre-`3fc6160a`: scale set `accepted=true`, so `while !accepted && mean_alpha` skipped mean entirely → mean freeze while cov drifted.
2. Tip `3fc6160a`: forced mean after scale, but mean trust region started at `0.1` and scale still ran first; remapped-LL-improving full mean steps were never proposed.

## Post-repair behavior (same inputs)

- mean-first recovery; mean alpha ladder starts at `1.0`
- selected: `iter=1;branch=mean_then_scale_accept;mean_alpha=1.000e0;scale_alpha=1.000e-1;ll=-7.3751378502e2;fixed_ll=-7.3751378502e2;scale_ll=-7.3428832892e2;scale_pd=true;mean=[-0.07459283003934392, -0.25210721473719855]`
- first recovery mean step uses full posterior-moment target under the same remapped LL / fixed_ll guards
- fixture `[0.65,-0.35]` remains recovery-test only (not an optimizer input)
