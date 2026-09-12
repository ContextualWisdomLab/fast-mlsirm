# Product and technical gap live refresh — 2026-09-12

Status: **Non-authoritative live supplement**  
Protected-product authority: `main@493326f2de49ea1704da0ded19868ed05d2fe00f`  
Central workflow authority: `ContextualWisdomLab/.github@691fb78932eff5fbe52db69077848134b0b4e053`  
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

## Central Required CodeQL prerequisite remains unresolved after replay-guard repair

`ContextualWisdomLab/.github#2051@558693e0333e48012beea142f739bc634b0674a7` and its child `#2056@69ae472562c93cc17674af5e2085a58947d3fab8` remain Draft/open. #2051 is 32 commits behind protected `.github/main@cb0872c9a20d5584703dffacca65c096fc034c6c`; #2056 and canonical successor #2040 diverge at `7fd571dbcdbae6acf29d8f4ee704d7ba6297e4db` with 24 and 178 unique commits respectively. Fresh source comparison verifies that #2040 preserves #2056's complete-failed-job-set invariant in evolved form: every supplied exact job must be a terminal failure and no terminal failed job may exist outside that supplied set before one run-wide rerun. The predecessors stay open until ordinary successor integration and hosted equivalence make retirement safe; source carryover is not stale approval or shipped evidence.

The canonical combined candidate advanced by an ordinary two-parent merge to `ContextualWisdomLab/.github#2040@798a5d4f2ceaf6ccc263c87b3f1d9f5e9f97bcbd`, tree `b3b8fa33b62eb10e8ddfad628ba077a3cfe05dfe`. Its parents are predecessor `6476b919d3febf79cc53e71d6d60f15d7e83ced4` and protected `.github/main@691fb78932eff5fbe52db69077848134b0b4e053`; the force=false update preserves both the full CodeQL/SARIF proposal and main's repository README standard. The sole conflict was the independent `CHANGELOG.md` prepend, resolved by retaining both entries. GitHub now reports the PR mergeable and zero commits behind.

Current-head OpenCode run `34684439092`, coverage job `103528902332`, supplied a new owner RED before coverage measurement: the protected `pr_head_replay_guard` saw intentional retirement of obsolete same-tree restamp tests in `tests/test_pr_review_merge_scheduler.py` but no added replacement test module. The repair moved the two positive no-mutation contracts into `tests/test_pr_review_merge_scheduler_source_neutral_commit_contract.py`. A transient Git-data upload commit `be25a767aa2e0d5cb79d6988853011bae16d4be3` truncated the large existing test blob; compare caught it immediately, no evidence was promoted from that head, and the ordinary child `6476b919...` restored the complete blob. Relative to the prior valid head, the final delta is exactly 46 lines moved from the existing test module and 60 lines added in the replacement module, with no production or workflow path changed.

The repaired predecessor tree had focused `9 passed / 325 deselected`, repository `3127 passed / 1 skipped / 21 subtests`, statement/branch/public-doc coverage 100%, production replay guard PASS, and a clean diff check. The reconciled exact tree `b3b8fa33...` then passed focused CodeQL/scheduler/README contracts (`392 passed / 15 subtests`) and the full repository suite (`3131 passed / 1 skipped / 36 subtests`), with scheduler-core statement/branch coverage and public-doc coverage at 100%, compileall PASS, and diff check clean. Fresh current-head runs `34686471029`, `34686471087`, `34686471109`, `34686471070`, `34686471090`, and `34686471227` are queued and are not promoted as GREEN. #2040 remains Draft/Proposed pending terminal exact-head Checks and independent current-head review. The earlier per-language protected-handler wake race remains an unresolved rollout dependency, not permission for a leaf repository to weaken Required CodeQL, synthesize compatibility statuses, create no-op commits, or copy central workflow logic.

## Landing authority

Protected `main` and the immutable release remain unchanged, so none of the lanes above is shipped. #1717 and #1816 stay Draft because their causal capability/evidence gaps are material. #1815 and #1819 have strong leaf acceptance evidence but are intentionally contained as Draft while their causal central Required CodeQL dependency remains RED. #1519 remains the single product-gap writer and preserves `docs/product-technical-gap-baseline.md` byte-for-byte; this dated supplement is additive evidence only.

No self-approval, force update, destructive rebase, routine administrator bypass, gate weakening, source-neutral retrigger, predecessor-success transfer, mutable foreign-owner dependency, skipped scientific evidence, or CPU result presented as GPU evidence is accepted as completion.