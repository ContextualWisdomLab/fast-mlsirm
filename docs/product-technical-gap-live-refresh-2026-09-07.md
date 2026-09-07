# Product and technical gap live refresh — 2026-09-07

Status: **Non-authoritative point-in-time supplement**  
Observed at: **2026-09-06T22:23:00Z**  
Protected-product basis: **`main@493326f2de49ea1704da0ded19868ed05d2fe00f`**  
Canonical historical baseline: **`docs/product-technical-gap-baseline.md`**  
Previous additive supplement: **`docs/product-technical-gap-live-refresh-2026-09-05.md`**  
Latest immutable release: **`v0.9.1`**, published 2026-08-26

This supplement records only evidence that changed after the preceding snapshot.
It does not replace the PRD, TRD, architecture, ADR, UML/ERD, Context Map,
requirements traceability, scientific evidence, or the protected historical
baseline. Open branches and their checks remain evidence, not shipped product
authority. Every merge or release decision must re-fetch the exact head, live
base, review threads, required checks, ruleset result, and active writer.

## Authority and ownership continuity

- Product scope and acceptance authority remain `docs/PRD.md` and
  `docs/TRD.md`.
- The repository boundary and Context Map remain `ARCHITECTURE.md` and the
  status-bearing ADR graph under `docs/adr/`.
- UML/ERD and requirements authority remain the linked document families in
  the protected baseline; this supplement creates no competing model.
- `fast-mlsirm` owns reusable, domain-neutral IRT/LSIRM/MLSIRM mathematical and
  Psychometrics kernels, true-parameter recovery, stable bindings, and release
  evidence. Result-affecting covariance, correlation, vector, linear, matrix,
  likelihood, estimation, scoring, uncertainty, and recovery arithmetic remain
  Rust/PyO3 owned.
- `ContextualWisdomLab/.github` owns reusable CI, review, security, and release
  orchestration. A leaf repository does not copy or bypass an immature central
  workflow; it waits behind the released contract or uses a bounded test double.

## Fresh inventory

GitHub search returned **64 open pull requests** and **201 open issues** for
`ContextualWisdomLab/fast-mlsirm` at the observation time. These counts are a
denominator for this snapshot, not a live invariant. Protected `main` and the
latest immutable release have not moved since the 2026-09-05 supplement.

## Buyer-visible gap and action status

| Gap / bounded context | Exact evidence | Action | Status |
| --- | --- | --- | --- |
| Central scheduler REST workflow identity | `ContextualWisdomLab/.github#1986` exact head `4604909a9b68cb29cda431d71bc0ed3d37f11af3`, protected base `c9052e607e5f3cc76e73207e7786b21500721b79`, two changed files | Preserve per-workflow parallelism while coalescing concurrent reads for one `(repository, workflow_id)`; treat a deleted workflow's HTTP 404 as an absent static identity while propagating other failures | **Ready, auto-merge armed behind protection.** RED reproduced 11 duplicate reads and 404 propagation. GREEN: 21 focused tests; scheduler aggregate 349 passed; repository 2,990 passed / 1 skipped / 21 subtests; coverage and public-doc checks 100%. Hosted exact-head security/review checks remain non-terminal and no current-head approval exists. |
| Actions queue measurement and workflow-waste evidence boundary | `ContextualWisdomLab/.github#1905` exact head `fcdb8dfe6951704cce688ec2e3756837b04bd71c`, two-parent reconciled on current #1903 `f4ff7f8c025c4d0a15145c3cd634d96c92326ec3`, one ledger path | Preserve the 35-workflow static observations while distinguishing a modeled admission ceiling from observed concurrent occupancy and a measured lower bound; continue capacity investigation and required-context-preserving graph repair in parallel | **Ready stacked evidence, not protected authority.** Current #1903 is an ancestor (`behind_by=0`); the contradicted global conclusions remain removed or bounded, and the residual literal `job\\namong` formatting defect is repaired. Exact-head baseline contracts are 5 passed and diff check is clean. Ready review admission was restored on the unchanged head at `2026-09-07T03:16:43Z`; CodeQL PR `34079111710`, Semgrep `34079111737`, and Security `34079111714` are queued. The predecessor-head CHANGES_REQUESTED review does not transfer; independent current-head review and ordinary stacked integration remain pending. |
| Review and CodeQL dispatch identity | `ContextualWisdomLab/.github#1929`; actor canary run `34069437294`; earlier cross-repository status failures `34017996201` and `34018021069` | Preserve actor=sender=one reviewed identity; prove a same-repository terminal canary and repair the separate cross-repository status credential without widening trust | **Actor admission repaired; issue remains open.** Live configuration now admits both `github-actions[bot]` and `opencode-agent[bot]`, and the canary passed actor validation before a later live-head mismatch. No same-repository terminal receipt is yet proven, while the cross-repository publication path still has HTTP 403 evidence. |
| CodeQL rerun recovery, live-base binding, terminal receipt, and SARIF evidence boundary | `ContextualWisdomLab/.github#1902` exact head `4b025af481f3a4fb0bdb4d400a7e055066a496a2`, tree `f0fa29d998727d9c8720cf2921611ff463667377`, nine CodeQL-owned paths | Require a trusted creator plus exact base/head/language/workflow/run receipt before consuming terminal status; ignore old-base status and redispatch once as `verdict=pending`; validate live/event base; require the same shard's SARIF upload outcome to be `success` before terminal publication or wake | **Ready for review admission; not merge-authorized.** A new CodeRabbit CWE-345 finding was reproduced and repaired. The requested old-base-only fixture proves pending redispatch; CodeRabbit confirmed the finding addressed and the sole thread is resolved. Exact-tree verification: 57 focused tests normally and 57 with `GITHUB_ACTIONS=true`; repository 3,005 passed / 1 skipped / 21 subtests; statement/branch coverage and public-doc coverage 100%; diff check clean. Local actionlint was unavailable, so predecessor evidence is not transferred. The Ready event admitted replacement runs `34077606247`, `34077606217`, `34077606226`, and `34077606207`, all queued/pending. Hosted GREEN and qualifying independent approval remain merge gates, not Ready prerequisites. No auto-merge action was taken. |
| Scheduler live-PR, Strix rerun identity, and lifecycle evidence preservation | `ContextualWisdomLab/.github#1999` exact head `64d19495095f42c292675dac9d7b73e8a6316d58`, tree `212153594cf4d90a9efb2e14526f408aa7634210`, stacked on `.github#1938@056226c56eff8c1aa01d29722f14c9820b97438d`, twelve scheduler/Strix/workflow-contract/doctoring paths | Require an explicitly open live PR, bind reruns to the verified failed Strix job, preserve executing same-head evidence across Draft/Ready/dispatch, wait for stale-run cleanup before launching the replacement provider, query/cancel central workflow runs at the run-owning repository while reading live PR state from the target repository, and retain protected-ref push coalescing with cancellation authority only for a newer push | **Ready for review admission; no auto-merge authorization.** Exact-head review found that the replacement provider could start before cleanup and that cleanup omitted stale `repository_dispatch` runs. RED reproduced both defects and the cross-repository ownership mismatch. GREEN is six focused lifecycle/cleanup tests, the focused Strix shell contract, repository 3,045 passed / 1 skipped / 21 subtests, statement/branch coverage 100% (`13,251` statements and `5,362` branches, zero miss/partial), and public-doc coverage 100%; `bash -n` and diff check are clean. The repair covers native and dispatched runs, separates target-repository PR reads from central run queries/cancellations, and gates provider start on cleanup success or skip. The stack is behind 0 and mergeable with zero unresolved threads; Ready was restored at `2026-09-07T05:39:30Z` only for review admission. Security `34087645347`, CodeQL PR `34087645423`, and Semgrep `34087645342` are queued, while push Semgrep `34087573459` is in progress; none is promoted to GREEN. Local actionlint was unavailable; hosted workflow validation and qualifying independent review remain merge gates. |
| Immutable release SBOM and provenance | `fast-mlsirm#1692` exact head `a6ac0f49d5123244fe89f26748a65f551ad9d514`, protected base `493326f2de49ea1704da0ded19868ed05d2fe00f`, 12 changed files | Restore an authenticated current-head CodeQL dispatch verdict at the central owner, obtain a qualifying approval on the unchanged current head, then ordinary merge and release verification | **Source-ready, control-plane blocked.** Repository CI, native CodeQL, Security Scan, Semgrep, mergeability, and all review threads are GREEN/resolved. `CodeQL PR` run `34020936743` fails closed because a rerun has no authenticated terminal verdict; predecessor approval does not transfer. |
| Rust-owned local-dependence public API | `fast-mlsirm#1748` exact head `ef2dd4baa11027c43fccc448a8eb07e4dca6e104`, protected base `493326f2de49ea1704da0ded19868ed05d2fe00f`, 14 changed files | Re-fetch current-head scientific recovery, public-contract, coverage, security, and independent-review evidence before merge | **Ready and mergeable; not yet revalidated in this supplement.** The prior supplement's older head and checks are historical only. |
| Rust covariance-standardization stack | `fast-mlsirm#1722` exact head `28b0305595107fd0ba21d7b27c1ac5db68ae8bf1`, direct protected base `main@493326f2de49ea1704da0ded19868ed05d2fe00f`, five changed files | Preserve the Rust production owner and reacquire exact-head numerical recovery, formatting, security, CodeQL, and independent-review evidence before ordinary integration | **Non-force reconciled, behind 0, partially GREEN.** The prior `b5a3a0c1` value was an older protected-main SHA, not a live stacked base. Current `main` was merged normally and a changelog fragment was added. Native CodeQL `34076849581` and CI `34076849582` are terminal success; Semgrep `34076849554`, CodeQL PR `34076849542`, ClusterFuzzLite `34076849579`, and Security `34076849578` remain queued. The local environment has no Cargo, so no separate local Rust test or formatting GREEN is claimed; predecessor evidence and approval do not transfer. |
| Product/technical gap evidence | `fast-mlsirm#1519`, draft single-writer branch `docs/refresh-product-gap-baseline-20260828` | Preserve the historical baseline and dated supplements; consolidate only after a reviewed proof that no PRD/TRD/UML/ERD/Context Map/scientific/release evidence is lost | **Active single writer.** This file is an additive delta on that branch, not a competing baseline writer. |

## Release and claim boundary

`v0.9.1` remains the latest immutable release. Neither a mergeable PR, a local
GREEN suite, queued hosted work, a resolved thread, nor an approval on an older
head is a release or GA claim. Technical GA still requires a bounded support
matrix, unchanged-head scientific recovery and cross-engine evidence where
applicable, stable API/artifact migration and rollback contracts, package/SBOM/
provenance evidence, security and operability gates, and ordinary protected
integration. Domain validation, high-stakes use, hosted identity, consent,
persistence, human decision policy, and buyer workflow validation remain owned
by the consuming product.

## Next safe sequence

1. Prove a same-repository terminal dispatch receipt under the repaired actor
   setting tracked by `.github#1929`, then repair and prove the separate
   cross-repository HTTP 403 status path without weakening actor=sender.
2. Let `.github#1902@4b025af4` and stacked `.github#1999@64d19495` acquire exact-head
   hosted checks and independent review; neither Ready transition is approval or
   merge authority, and no auto-merge change is recorded here.
3. Revalidate `fast-mlsirm#1692` on its unchanged head after the owner repair,
   merge ordinarily, and
   produce immutable release evidence before changing the released-version
   claim.
4. Revalidate `fast-mlsirm#1722@28b03055` on its direct protected-main base,
   including Cargo/Rust formatting and numerical recovery evidence, before
   continuing the Rust numerical stack in dependency order.
