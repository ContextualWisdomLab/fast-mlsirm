# Product and technical gap live refresh — 2026-09-05

Status: **Non-authoritative live supplement**  
Protected-product basis: `main@493326f2de49ea1704da0ded19868ed05d2fe00f`  
Canonical historical baseline: `docs/product-technical-gap-baseline.md`  
Previous additive supplement: `docs/product-technical-gap-live-refresh-2026-09-04.md`  
Latest immutable release: `v0.9.1` (published 2026-08-26, immutable)

This file updates only live evidence that moved after the 2026-09-04 supplement. The protected 1,036-line baseline and the preceding supplement remain preserved; neither is rewritten or treated as disposable history. A capability becomes product authority only after integration into protected `main` and terminal applicable scientific, package, coverage, security, review, SBOM/provenance and release evidence on one unchanged exact head.

## Ownership boundary

`fast-mlsirm` remains the canonical reusable psychometric numerical owner for LSIRM/MLSIRM/IRT/generalized-dependence kernels, true-parameter recovery and stable public bindings. Result-affecting likelihood, estimation, scoring, uncertainty, covariance/correlation, vector/linear/matrix and recovery arithmetic remain Rust/PyO3 owned. Python remains validation, provenance sealing, marshalling, orchestration, reporting and explicit reference/parity evidence.

TEPP owns temporal/event semantics and composition. `contextual-orchestrator` owns provider/model routing and LLM orchestration. Foreign domain truth is consumed only through released/versioned contracts or explicit ACLs; source copying, mutable sibling-head dependencies and cross-service SQL are not product authority.

## Exact live owner lanes

| Gap | Exact owner state | Remaining acceptance |
| --- | --- | --- |
| Protected GPU merge gate + repository PR lifecycle | #1717 Ready `fbe1262050bf00e6bd71b6709fca81902ae21a52` | Forward reconciliation preserves explicit Ubuntu 24.04 runner identity and the protected `python -> [python-matrix, gpu-smoke]` dependency while also adopting #1749's repository/PR concurrency, Draft/Ready/closed events, PR-only cancellation and inactive-Draft suppression. CI `33924705580` materialized fuzz, rust, package, Python 3.12, gpu-smoke and Python 3.14 but they remain queued without runner execution. Terminal exact-head GREEN and independent approval remain required. |
| Local-dependence stable public API | #1748 Ready `e7e1e71be6d90d11f6e1f604235c447bfa75cdbb` | Existing Rust-owned Chen–Thissen X2/G2 arithmetic is exposed through a hardened Python/package-root boundary. Input controls and native result envelope/cardinality/finiteness/package-ownership are replayed without moving psychometric arithmetic out of Rust. Ready created fresh exact-head CI/CFLite/security/CodeQL/Semgrep execution, but current jobs remain queued. No universal item-removal cutoff is claimed. |
| Marginal reduction reproducibility | #1742 Draft `dbb6a9bf74e940280fc5b0c247469b7850534709` | Test-only successor pins three invalid floating-point reassociation families: ordinary split-dot one-ULP drift, saturated-logit `NaN` from the rejected algebraic split, and row-wise `einsum` squared-distance one-ULP drift. Production estimator/objective/distance code remains protected-main behavior. Any future hot-path optimization needs profiling plus deterministic CPU-f64 and realistic recovery/parity evidence and should prefer the Rust numerical owner. |
| Oblimax deterministic CPU-f64 reference | #1736 Draft `44806471463dda11ed251c4e43c3f9e9e0f7293a`; issue #1747 | Deterministic log/power routing, ratio-before-log cancellation repair, exact power-of-two range conditioning, public optimizer scale contract and exact stationary-manifold fixes are numerical evidence. Scientific completion still requires supported-target bit parity plus VV-SCI-006 known-population recovery: globally sign/permutation-aligned Tucker congruence, loading/target RMSE where identified, bootstrap/split-sample stability, basin support/entropy, criterion-selection frequency and factor-correlation/degeneracy diagnostics. |
| v0.9.2 immutable release | #1471 Draft `d6edc8ea83d8bd0b0840786ca4e8974623560b1f` | Current-main ancestry and Draft provenance are repaired, but release serialization remains RED. `CHANGELOG.md` still carries the superseded managed Unreleased block and historical `[0.9.2] - 2026-08-27` while authoritative fragments contain later protected-main evidence. The exact release head must atomically recut the changelog, preserve the original 17 folded deltas, assign the actual release date only at cut time, then reacquire version/lock/test/security/package/SBOM/provenance/reproducibility/rollback/review evidence. |
| Item-bank evidence-reference provenance child | #1476 Draft `f3c66f0e9f7788e0e20e4f3ef4cbbe5f110fa811`, stacked on #1471 | Child remains intentionally un-restacked while #1471 is source-RED. Its three valid provenance/error-boundary files remain intact. After parent recut, non-force restack onto that exact parent and regenerate hosted/current-head evidence. |
| Product/technical gap evidence preservation | #1519 single writer | Preserve the historical baseline plus dated additive supplements. Do not replace evidence-rich history with a short live inventory. Consolidation is allowed only after a reviewed diff proves no valid PRD/TRD/UML/research/release/buyer/accessibility/traceability evidence is lost. |

## Current merge and workflow authority

Protected `main@493326f2de49ea1704da0ded19868ed05d2fe00f` requires `Analyze (actions)`, `close-empty`, `scan-pr-queue`, `dependency-review`, `osv-scan`, `trivy-fs`, `scorecard`, `required-workflow-bootstrap`, `coverage-evidence`, `opencode-review`, `python`, `rust`, `package` and `fuzz` contexts. #1749 added repository/PR-scoped concurrency and inactive-Draft lifecycle control without removing the local CodeQL producer.

Central workflow authority is `.github/main@f43dcb884be5a0efc61611b5c8cb83c4c7735995`. Current #1717 and #1748 canaries distinguish working workflow materialization/lifecycle from unresolved runner admission: their substantive exact heads materialize jobs, but those jobs remain queued without runner/group execution. That is not terminal GREEN and is not a reason to weaken gates, copy central workflows locally or manufacture no-op commits.

## Release/scientific claim rule

No Draft or Ready branch above is a shipped capability. Predecessor GREEN, source-level RED/GREEN, queued workflow creation and resolved review threads are evidence inputs, not substitutes for unchanged-current-head terminal hosted success. Scientific/release claims require their stated recovery/parity/provenance gates in addition to ordinary CI/security and a qualifying independent current-head approval. No self-approval, bypass, force update, destructive rebase, gate weakening, skip/xfail success accounting or predecessor-success transfer is authorized.
