# Product and technical gap live refresh — 2026-09-07

Status: **Non-authoritative point-in-time supplement**  
Observed at: **`2026-09-08T13:01:15Z`**  
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
| CodeQL rerun recovery, live-base liveness, exact-run settlement, and receipt identity | `ContextualWisdomLab/.github#1902` exact head `e0924260c2105b49e8840701ce8509d765125b0f`, tree `b41c05ff2410c1ebc7799e491a7da04483ee55e7`, protected base `main@7fd571dbcdbae6acf29d8f4ee704d7ba6297e4db`, twelve CodeQL-owner paths | Capture and revalidate one live base across the attempt; allow only a compare-proven same-ref strict forward advance to trigger a whole exact-run rerun; distinguish absent evidence from multiple complete producers and stop ambiguity before credential acquisition or dispatch | **Source repaired and Ready for review admission; hosted gates pending.** Force=false descendants preserve concurrent gate-proof, late-base RED, and handler-window repair. RED reproduced late base-advance rejection plus three missing-telemetry ambiguity cases. GREEN: ambiguity focused 3 passed; repository 3,086 passed / 1 skipped / 21 subtests; statement `13,194/13,194`, branch `5,330/5,330`, total and public-doc coverage 100%; compileall and diff check clean. Multiple complete receipt/direct candidates emit exact run-ID/state telemetry and cause no token request or dispatch. Protected-base comparison is 62 ahead / 0 behind, mergeable, with zero unresolved threads. Ready was restored again at `2026-09-08T10:21:04Z` after a concurrent `10:16:24Z` Draft event with no GitHub App attribution, new review, unresolved thread, or source finding; further same-head toggling is a lifecycle/single-writer defect, not a reason for repeated state churn. CodeQL PR `34214980549`, SAST Semgrep `34214980613`, Python Security `34214980554`, and Security Scan `34214980560` are queued. Agent Review Runtime Quality `34214227385` is terminal success; the preceding Ready-event runs were cancelled and are not promoted. Hosted checks and qualifying independent current-head review remain merge gates; local actionlint is unavailable. Evidence: `.github#1902` comment `5583516600`. |
| Scheduler live-PR, Strix rerun identity, and lifecycle evidence preservation | `ContextualWisdomLab/.github#1999` exact head `64d19495095f42c292675dac9d7b73e8a6316d58`, tree `212153594cf4d90a9efb2e14526f408aa7634210`, stacked on `.github#1938@056226c56eff8c1aa01d29722f14c9820b97438d`, twelve scheduler/Strix/workflow-contract/doctoring paths | Require an explicitly open live PR, bind reruns to the verified failed Strix job, preserve executing same-head evidence across Draft/Ready/dispatch, wait for stale-run cleanup before launching the replacement provider, query/cancel central workflow runs at the run-owning repository while reading live PR state from the target repository, and retain protected-ref push coalescing with cancellation authority only for a newer push | **Ready for review admission; no auto-merge authorization.** Exact-head review found that the replacement provider could start before cleanup and that cleanup omitted stale `repository_dispatch` runs. RED reproduced both defects and the cross-repository ownership mismatch. GREEN is six focused lifecycle/cleanup tests, the focused Strix shell contract, repository 3,045 passed / 1 skipped / 21 subtests, statement/branch coverage 100% (`13,251` statements and `5,362` branches, zero miss/partial), and public-doc coverage 100%; `bash -n` and diff check are clean. The repair covers native and dispatched runs, separates target-repository PR reads from central run queries/cancellations, and gates provider start on cleanup success or skip. The stack is behind 0 and mergeable with zero unresolved threads; Ready was restored at `2026-09-07T05:39:30Z` only for review admission. Ready-event Security `34087645347`, CodeQL PR `34087645423`, and Semgrep `34087645342` are queued; the earlier push runs, including Semgrep `34087573459`, were cancelled after the Ready event, and none is promoted to GREEN. Local actionlint was unavailable; hosted workflow validation and qualifying independent review remain merge gates. |
| Immutable release SBOM and provenance | `fast-mlsirm#1692` exact head `a6ac0f49d5123244fe89f26748a65f551ad9d514`, protected base `493326f2de49ea1704da0ded19868ed05d2fe00f`, 12 changed files | Restore an authenticated current-head CodeQL dispatch verdict at the central owner, obtain a qualifying approval on the unchanged current head, then ordinary merge and release verification | **Source-ready, control-plane blocked.** Repository CI, native CodeQL, Security Scan, Semgrep, mergeability, and all review threads are GREEN/resolved. `CodeQL PR` run `34020936743` fails closed because a rerun has no authenticated terminal verdict; predecessor approval does not transfer. |
| Rust-owned local-dependence public API | `fast-mlsirm#1748` exact head `ef2dd4baa11027c43fccc448a8eb07e4dca6e104`, protected base `493326f2de49ea1704da0ded19868ed05d2fe00f`, 14 changed files | Re-fetch current-head scientific recovery, public-contract, coverage, security, and independent-review evidence before merge | **Ready and mergeable; not yet revalidated in this supplement.** The prior supplement's older head and checks are historical only. |
| Rust covariance-standardization stack | `fast-mlsirm#1722` exact head `28b0305595107fd0ba21d7b27c1ac5db68ae8bf1`, direct protected base `main@493326f2de49ea1704da0ded19868ed05d2fe00f`, five changed files | Preserve the Rust production owner and reacquire exact-head numerical recovery, formatting, security, CodeQL, and independent-review evidence before ordinary integration | **Non-force reconciled, behind 0, partially GREEN.** The prior `b5a3a0c1` value was an older protected-main SHA, not a live stacked base. Current `main` was merged normally and a changelog fragment was added. Native CodeQL `34076849581` and CI `34076849582` are terminal success; Semgrep `34076849554`, CodeQL PR `34076849542`, ClusterFuzzLite `34076849579`, and Security `34076849578` remain queued. The local environment has no Cargo, so no separate local Rust test or formatting GREEN is claimed; predecessor evidence and approval do not transfer. |
| Product/technical gap evidence | `fast-mlsirm#1519`, Ready single-writer branch `docs/refresh-product-gap-baseline-20260828` | Preserve the historical baseline and dated supplements; consolidate only after a reviewed proof that no PRD/TRD/UML/ERD/Context Map/scientific/release evidence is lost | **Active single writer.** This file is an additive delta on that branch, not a competing baseline writer. |

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

1. Let `.github#1902@e0924260` prove its base-advance whole-attempt recovery and exact required-run-bound receipt/direct-evidence path in hosted execution, then repair and prove the separate cross-repository HTTP
   403 publication path tracked by `.github#1929` without weakening identity.
2. Let `.github#1902@e0924260` and stacked `.github#1999@64d19495` acquire
   exact-head hosted checks and independent review; keep `.github#2032` preserved
   until #1902's hosted equivalence is proven. Ready is not approval or merge
   authority, and no auto-merge change is recorded here.
3. Revalidate `fast-mlsirm#1692` on its unchanged head after the owner repair,
   merge ordinarily, and
   produce immutable release evidence before changing the released-version
   claim.
4. Revalidate `fast-mlsirm#1722@28b03055` on its direct protected-main base,
   including Cargo/Rust formatting and numerical recovery evidence, before
   continuing the Rust numerical stack in dependency order.

## 2026-09-08 central handler prerequisite correction

`ContextualWisdomLab/.github#2040` is the canonical handler prerequisite for the
current CodeQL recovery stack. Its exact head
`d93a78ab4262c5228af7eda258ee0af58b880de7` (tree
`bc5e85327c4a531c935091c623e68ec8868e539b`) is 19 commits ahead and 0
behind protected `main@7fd571dbcdbae6acf29d8f4ee704d7ba6297e4db`, changes
seven intended paths, is mergeable, and has zero unresolved review threads.
The handler accepts one nested or legacy rerun envelope, rejects coexistence
and ambiguity, binds the request to an immutable producer workflow source,
keeps matrix scans at `actions: read`, and gives one non-matrix settlement
owner `actions: write`. That owner revalidates the open PR, exact
head/base/ref, required run/jobs, handler job steps, nonexpired nonempty SARIF
artifacts, and unrelated-failure absence before one run-wide recovery request.

RED `e0800adf0` required source-bound settlement evidence; GREEN is
`d93a78ab4262c5228af7eda258ee0af58b880de7`. Exact-tree verification is 45
focused tests, 3,010 passed / 1 skipped / 21 subtests repository-wide,
statement and branch coverage 100%, public-doc coverage 100%, and clean diff
check. Local `actionlint` is unavailable. Ready was restored at
`2026-09-08T13:01:15Z` solely for review admission. Its automatically created
Ready-generation runs are CodeQL PR `34229436485`, Python Security
`34229436817`, SAST Semgrep `34229436541`, and Security Scan
`34229436635`; none was promoted to GREEN in this snapshot. A qualifying
independent current-head approval is also absent. Evidence:
`.github#2040` comment `5585550769`.

The predecessor-head protected-handler run `34228191430` proves the
protected-main sibling-wake race but is neither current-head hosted GREEN nor
approval. `.github#1902` remains Draft until #2040 integrates ordinarily and
#1902 can be reconciled non-destructively. No bypass, manual rerun, synthetic
status, auto-merge authorization, or predecessor-evidence transfer is recorded.

## 2026-09-08 canonical combined handler correction

The preceding `d93a78ab…` snapshot is superseded by
`ContextualWisdomLab/.github#2040@a22dd5b7108698f5ae0c8dc0541c4dcc7b61abae`
(tree `d1ebad5966d83e658a8f44bf806af4b0652aedd0`). The exact head is an
ordinary two-parent child of prior canonical #2040
`9054b5e664d1540736c6b83a7c4a2bb9399651a3` and
`ContextualWisdomLab/.github#2044@3720dd853fe399fd453e093a90944b2b0e78a8e6`.
This makes #2044's valid dual-identity repair, tests, documentation, and
requirements explicit in the combined successor's ancestry while all
predecessors remain open Proposed evidence.

The exact tree rejects conflicting ref, conflicting SHA, and either partial
nested/legacy representation with separate executable fixtures. A fresh full
run first exposed the inherited scheduler invariant gap at
`pr_review_merge_scheduler_core.py:497`: 3,026 tests passed, but one uncovered
statement/branch left total coverage below the 100% contract. The minimal
fixture then produced focused 60 passed, repository-wide 3,027 passed / 1
skipped / 21 subtests, statement and branch coverage 100%, public-doc coverage
100%, and clean diff check. Protected-base comparison is 56 ahead / 0 behind,
17 bounded paths, mergeable, with zero unresolved threads at publication.

#2040 remains Ready only for review admission. Fresh exact-head runs are Python
Security `34237937741`, Security Scan `34237937787`, Agent Review Runtime
Quality `34237938200`, SAST Semgrep `34237937609`, and CodeQL PR
`34237937656`; all were non-terminal in this snapshot. No qualifying
current-head independent approval exists, so ordinary merge remains blocked.
Evidence: `.github#2040` comment `5586694059`. No merge, auto-merge
authorization, approval, protection bypass, manual rerun, synthetic status,
empty push, force push, destructive rebase, or predecessor-evidence transfer
is recorded.

## 2026-09-08 central Gitleaks stack reconciliation

Canonical CodeQL successor `ContextualWisdomLab/.github#2040` advanced
concurrently from `a22dd5b7…` to exact head
`d4a95632af9031d7a40d3cab7e78c04f87044db4`. The two-commit delta suppresses
status publication after a superseded-base revalidation and adds a bounded
dual-context migration bridge with an explicit removal condition. Its Python
Security `34239914762`, Security Scan `34239914829`, Runtime Quality
`34239914795`, and Semgrep `34239914781` runs succeeded. CodeQL PR
`34239914870` failed after the protected producer dispatched pending shards;
handler run `34241333744` remained queued in this snapshot. #2040 is therefore
Draft/Proposed and has no qualifying current-head approval.

The dependent Gitleaks owner `ContextualWisdomLab/.github#2041` was first
reconciled to #2040@`a22dd5b7…`, then non-force advanced again when the
prerequisite moved. Its final exact head
`c51aae180f621d709163d295f1e9113671bc9585` (tree
`ea330d5467e05a826acd0545ffd00ce742bc3c7c`) is an ordinary two-parent child
of prior #2041 `50ff203027345bc6337d259bcec511cd98ab554b` and current #2040
`d4a95632af9031d7a40d3cab7e78c04f87044db4`. Relative to #2040 it is 16
commits ahead / 0 behind, mergeable, and retains exactly four effective
Gitleaks-boundary paths. The append-only Gap merge preserves both owner evidence
sections.

Exact merged-tree verification is 61 focused CodeQL/Gitleaks tests and
repository-wide 3,032 passed / 1 skipped / 21 subtests, with statement and
branch coverage 100%, public-doc coverage 100%, and clean diff check. Fresh
#2041 runs are Runtime Quality `34243109706`, Security Scan `34243109766`,
Semgrep `34243109758`, Python Security `34243109798`, and CodeQL PR
`34243109771`; all were non-terminal at publication. Evidence:
`.github#2041` comment `5587386220`. #2041 remains Draft/Proposed until the
prerequisite integrates and fresh exact-head checks plus qualifying independent
review are satisfied. Predecessor GREEN is non-authorizing.

No lifecycle toggle, merge, approval, auto-merge authorization, protection
bypass, manual rerun, empty push, force push, destructive rebase, review
dismissal, or predecessor-evidence transfer is recorded.
