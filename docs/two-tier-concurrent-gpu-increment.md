# Concurrent two-tier GPU item fitting increment

`mlsirm_core::two_tier_grm::concurrent::fit_concurrent_gpu` is an explicit native
concurrent shared/free-item path. It accepts arbitrary primary loading masks,
one specific loading per item, group IDs, an item-common mask and one explicit
fixed/free Gaussian mean/SD constraint per group and factor. Group0 fixes every
mean0 and SD1; primary correlation is exactly identity in every group. A common
item is estimated once using the concatenated group-specific node/count rows,
then copied bit-identically to each group. A free item is updated separately in
each group. This does not freeze the reference bank as FIPC.

Cai (2010, pp.586–590, Eqs.2,4–6,11–16; pp.608–609, Appendix A) supplies the
normal two-tier factorization, GRM and complete-data item/Gaussian criteria.
The independent-group sum with equality-constrained common item parameters is
implemented here as an extension; Cai's empirical presentation is single-group
(p.606). Affine Gauss–Hermite nodes, initialization, strict Newton controls and
convergence rules are implementation choices, not paper-prescribed thresholds.
The caller owns quadrature and model identification. Direct common loading
coverage and a fitted anchor-location column-rank check are necessary screens,
not full Fisher-information or variance-identification evidence.

## Execution and stopping

The existing `two_tier_gpu_item` adapter executes item criterion, gradient,
Hessian, solve, search and updates with unchanged gradient/step1e-9. GPU reduced
posterior/count/specific moments are required; CPU optimizer fallback is absent.
All attempted starts retain successful group-dispatch identities and partial
item-stage receipts, errors and initial/terminal E sweeps. A previous bounded
Newton sweep cannot certify convergence. Later stationary sweeps can recover.
Convergence also requires nondecreasing native f64 scalar likelihood and small
full item/mean/log-SD displacement. Budget exhaustion remains nonconvergence.

This is **hybrid**, not full-GPU: CPU builds probability tables, recomputes the
scalar f64 likelihood, reduces returned primary posteriors and updates Gaussian
parameters. Entirely unobserved specific blocks contribute their finite-GH
prior moments on CPU, separately counted. These CPU phases are explicit and do
not increment GPU-stage receipts. The shared E-step uses binary32, while item
Newton uses its existing two-f32 representation. Stopping is not accuracy.

## Verified and open scope

Small native synthetic controls cover mixed common/free items, nonstandard
fixed priors, free primary and specific means/SDs, missing item/block responses,
strict bounded-update refusal and later recovery, and the selected six-factor
G+4+W binary loading pattern. Tests use deliberately small grids as control-flow
fixtures. The four-category G+4+W fixture returns after the reviewed shader
repair in 274.327177 seconds within its 360-second bound, with one EM sweep
and nonconvergence. The earlier timeout and binary control remain separate
historical evidence (shader_independent_review/REVIEW.md, 2026-10-27).
No selected polytomous calibration, high-node recovery or private fit is accepted.

Python binding, mask/DIF lifecycle integration, same-fit M2/FIFA/information,
scoring with nonzero specific means, study focal W/specific decisions, full-GPU
CPU-phase replacement, and high-grid/scientific acceptance remain composition
work. The ordinary Python/single-group entrypoint still selects CPU; only its
internal explicit-device seam and the new native concurrent API consume GPU
item updates. Existing FIPC remains a separate path.

## Reference

Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. *Psychometrika, 75*(4), 581–612.
https://doi.org/10.1007/s11336-010-9178-0

The cited sections are read from the retained full text. Copyrighted PDF bytes
are not redistributed.
