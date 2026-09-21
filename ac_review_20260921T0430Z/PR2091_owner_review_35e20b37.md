# fast-mlsirm #2091 — fmls owner technical review at 35e20b37 (2026-09-21 ~09:35Z)

Scope: `predict_bifactor_expected_total_score(fit, theta, q_specific, *, group=None)` plus MG support in `check_bifactor_expected_total_score_monotonicity`. The diff is 5 files (+603/−46), with merge-base == origin/main `99c228a8`.

This is not a formal GitHub approval: the author account is shared, so independent review is routed via CO. It is not CI evidence either; 231d owns CI.

Verified from source:
- **Metric:** MG item parameters and `theta_g_eap` are on the common reference metric (E-step nodes `general_mean[g] + general_sd[g]·z`). Pointwise E[T|θ_G] correctly applies no per-group θ transform.
- **Group selection:** the group axis is selected, never flattened. `group=None` is admitted only if all groups' `a_general`/`a_specific`/`threshold` rows are exactly equal (true for `anchor_mask=None` by construction), so no silent group-0 pick happens for free items.
- **Specific factors:** they are integrated over N(0,1) GH nodes. A non-unit `specific_sd` (estimate_specific_vars=True) raises instead of approximating, because the fit carries no `specific_map`.
- **Kernel:** it is the previous monotonicity-check kernel moved verbatim. The check now delegates to the predictor, so the two cannot disagree.
- **Input guards:** theta is a finite non-empty 1-D array; `q_specific` is in 1..=4096 with no default (#1929 rule).

Research consumer fit (W, pkl `d74eb91a`): 3 groups, anchor None, `specific_sd` all 1 (estimate_specific_vars=False), so `group=None` is valid. The #2091 session reports max abs 0.0 vs the grid path over 1020 persons.

No blocking defect found. Remaining: independent review, hosted CI (231d), merge, release cut, research pin bump.
