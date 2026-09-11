# Product and technical gap live refresh — 2026-09-12

Status: **Non-authoritative live supplement**  
Protected-product authority: `main@493326f2de49ea1704da0ded19868ed05d2fe00f`  
Central workflow authority: `ContextualWisdomLab/.github@cb0872c9a20d5584703dffacca65c096fc034c6c`  
Canonical historical baseline: `docs/product-technical-gap-baseline.md`  
Latest immutable product release: `v0.9.1` (published 2026-08-26)

This supplement records only live movement after the 2026-09-11 refresh. It does not promote Draft work, mutable PR heads, predecessor checks, or foreign-owner proposals to shipped status. The historical baseline remains unchanged.

## GPU compute admission is now a capacity mismatch, not an acquisition failure

`fast-mlsirm#1717@0b31640928e07f4362ce27dad3d310e630ab1b5d` is open, mergeable and Draft. The controlled Ubuntu runner can acquire image-local SwiftShader and expose a Vulkan adapter, but the adapter reports `max_storage_buffers_per_shader_stage = 10` while the current marginal GPU topology requires at least 18 storage buffers per shader stage. Exact-head CI therefore fails closed in `gpu-smoke` at the capability invariant rather than silently accepting CPU fallback.

The environment-specific capacity probe has been moved out of the ordinary Rust test suite and into `crates/mlsirm-core/examples/gpu_adapter_capacity.rs`; only the dedicated GPU lane executes it. The remaining product gap is causal: either provide a reproducible adapter meeting the existing resource contract or redesign the marginal GPU resource topology and re-establish CPU-f64 parity, realistic performance and recovery evidence. Lowering the capacity contract, accepting skipped GPU evidence, or reporting CPU fallback as GPU execution remains inadmissible.

## Correctly-rounded binary64 mean remains the canonical reusable numerical proposal

`fast-mlsirm#1816@432765ccf633c9802e0f796ceeb4d6d572059acf` is open, mergeable and Draft. The branch owns the domain-neutral `fast_mlsirm.binary64_mean@1.0.0` contract: finite binary64 inputs are accumulated exactly as signed integer multiples of `2^-1074`, divided by the original cardinality, and rounded once with round-to-nearest, ties-to-even. The fixed 34×`u64` magnitude keeps the proved capacity bound for supported `usize::BITS <= 64` without imposing an arbitrary scientific sample ceiling.

The current suite includes the TEPP half-ULP counterexample, ordinary mixed-sign cancellation, MAX/subnormal residue, exact-zero versus signed underflow-zero, subnormal/normal transitions, ties-to-even and binade carry, same-sign `f64::MAX`, permutation invariance, non-finite rejection, and a deterministic 10,000-case independent subnormal integer/rational oracle. TEPP remains a consumer only after protected integration and immutable release; it must not copy source or depend on this mutable head.

Landing is still gated by owner-path evidence rather than a known numerical defect: repository GPU capability remains owned by #1717; Required CodeQL still has a central lifecycle failure; authoritative exact-head Rust line/branch evidence remains a central coverage requirement; and review/security infrastructure failures must be replayed rather than treated as clean results.

## DIMTEST release-build ownership repair is leaf-complete but centrally blocked

`fast-mlsirm#1815@7c2c2968f62496fa313e27682ae3ae46d2493b23` remains open, mergeable and Draft. The repair keeps `DimtestGroupDiag.{jk, mean, v, u, mu4, delta4, s2}` only in test builds while preserving the independent Nandakumar–Stout oracle and leaving production DIMTEST arithmetic and the public `DimtestResult` unchanged.

On the unchanged exact head, ordinary CI, package, Rust/PyO3, fuzz, explicit non-skipped GPU parity, repository CodeQL, Security Scan, Semgrep and ClusterFuzzLite are terminal GREEN, and Noema submitted a current-head `APPROVED` review with no blocking product finding. Normal merge remains prohibited because Required CodeQL PR is RED in the central producer-after-consumer lifecycle. Draft containment avoids source-neutral re-admission churn while that owner-path defect remains unchanged; no leaf status synthesis, no-op retrigger or copied central workflow is justified.

## ADR reservation governance now has exact-head leaf acceptance evidence

`fast-mlsirm#1819@50be41e202269931e091c7e8e0c036998ef64a6e` implements the repository-visible ADR allocation rule that coordinates protected numbers with still-valid live reservations. The current patch documents earliest-valid collision ownership, next-free allocation for later colliders in PR creation order, ordinary-forward renumbering, and the protected-tree uniqueness invariant; it also records the standards basis while correctly labelling GitHub reservation ordering as repository-local policy rather than an ISO/IEC/IEEE mandate.

Current-head Security Scan, repository CodeQL, Semgrep and ordinary CI are terminal GREEN. Noema submitted an exact-head `APPROVED` review with no blocking finding. Required CodeQL PR remains the sole known landing failure on this head and reproduces the central producer-after-consumer lifecycle defect. Coordination issue #1817 therefore remains open until this governance contract reaches protected main. `#1603@acf1891469f2eba456107584f692d4dd6280c1ed` owns the governed-rater decision now reserved as ADR-0030, and `#1694@9ba4e018c1ba359fc72e515fef2eb466e17c48b3` owns the Rust registry/distribution decision now reserved as ADR-0031; both remain Draft after ordinary-forward identity repair rather than being closed or rewritten.

## Central Required CodeQL prerequisite remains unresolved after protected-main reconciliation

`ContextualWisdomLab/.github#2051@558693e0333e48012beea142f739bc634b0674a7` remains Draft and explicitly requires a versioned, backward-compatible protected handler prerequisite before client cutover. Its distinct partial-shard/base-binding contract delta is still valid and has not been proven completely inherited by a successor; it therefore remains open and must itself adopt current protected authority before any exact-head acceptance claim.

The broad combined candidate `ContextualWisdomLab/.github#2040` has now ordinary/non-force reconciled protected `.github/main@cb0872c9a20d5584703dffacca65c096fc034c6c`. Current exact head `3b2de64c2c4c95c56d2f5099a480a0825304d038` has tree `eb8d6351193519afca23d448f70b9ab59f5eb5a4`; fresh comparison is 145 commits ahead and 0 behind with merge base exactly protected main. A concurrent writer produced the same union tree independently prepared during reconciliation, so that valid merge-forward was adopted rather than rewritten or force-replaced.

The reconciliation removed the stale-base defect but not the central bootstrap lifecycle defect. On the new exact head, Required CodeQL still lets python receiver `103362319731` and actions receiver `103362317835` fail enforcement before dispatch producer `103362504390` starts and later succeeds. #2040 therefore remains Draft while current-head checks and independent review are reacquired. The next causal repair remains a landable backward-compatible/versioned protected handler path; leaf repositories must not work around it by weakening Required CodeQL, synthesizing compatibility statuses, or copying central workflow logic.

## Landing authority

Protected `main` and the immutable release remain unchanged, so none of the lanes above is shipped. #1717 and #1816 stay Draft because their causal capability/evidence gaps are material. #1815 and #1819 have strong leaf acceptance evidence but are intentionally contained as Draft while their causal central Required CodeQL dependency remains RED. #1519 remains the single product-gap writer and preserves `docs/product-technical-gap-baseline.md` byte-for-byte; this dated supplement is additive evidence only.

No self-approval, force update, destructive rebase, routine administrator bypass, gate weakening, source-neutral retrigger, predecessor-success transfer, mutable foreign-owner dependency, skipped scientific evidence, or CPU result presented as GPU evidence is accepted as completion.