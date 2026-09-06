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
| Review and CodeQL dispatch identity | `ContextualWisdomLab/.github#1929`; `#1986@4604909a9b68cb29cda431d71bc0ed3d37f11af3` successfully created dispatch runs `34066603914` and `34066634411` | Preserve actor=sender validation; prove both runs reach terminal same-repository status publication, then prove the cross-repository credential path independently | **Identity admission has advanced; issue remains open.** The new dispatches passed request creation but remain queued. This supersedes the earlier “blocked at the actor allowlist” status. Cross-repository commit-status publication previously received HTTP 403 and remains unproven. |
| CodeQL rerun recovery | `ContextualWisdomLab/.github#1902` exact head `3549de595c9d7615b253e03fc18ee06f34d3d294`, tree `08661ee1e65295e07a69620292a764e0f9cb15e9`, five CodeQL-owned paths | Use complete paginated status history and trusted creator/context identity; if no terminal verdict exists, perform one bounded redispatch instead of treating `run_attempt` as a receipt | **Ready, ordinary auto-merge armed.** Focused 12 passed; repository 2,979 passed / 1 skipped / 21 subtests; coverage and public-doc checks 100%. Hosted exact-head checks and independent review remain non-terminal. |
| Scheduler live-PR and Strix rerun identity | `ContextualWisdomLab/.github#1999` exact head `ed8ab26b53792386e4578fcfbd3bdf1632cff54c`, tree `9619c391252b95b1118f2d81ed60c445e307bdbf`, seven scheduler-owned paths | Require an explicitly open live PR and bind reruns to the verified failed Strix job identity | **Ready, ordinary auto-merge armed.** This is the complete successor carryover of valid scheduler/Strix commits formerly mixed into #1902. Focused 394 passed; repository 3,039 passed / 1 skipped / 21 subtests; coverage and public-doc checks 100%. |
| Immutable release SBOM and provenance | `fast-mlsirm#1692` exact head `a6ac0f49d5123244fe89f26748a65f551ad9d514`, protected base `493326f2de49ea1704da0ded19868ed05d2fe00f`, 12 changed files | Restore an authenticated current-head CodeQL dispatch verdict at the central owner, obtain a qualifying approval on the unchanged current head, then ordinary merge and release verification | **Source-ready, control-plane blocked.** Repository CI, native CodeQL, Security Scan, Semgrep, mergeability, and all review threads are GREEN/resolved. `CodeQL PR` run `34020936743` fails closed because a rerun has no authenticated terminal verdict; predecessor approval does not transfer. |
| Rust-owned local-dependence public API | `fast-mlsirm#1748` exact head `ef2dd4baa11027c43fccc448a8eb07e4dca6e104`, protected base `493326f2de49ea1704da0ded19868ed05d2fe00f`, 14 changed files | Re-fetch current-head scientific recovery, public-contract, coverage, security, and independent-review evidence before merge | **Ready and mergeable; not yet revalidated in this supplement.** The prior supplement's older head and checks are historical only. |
| Rust covariance-standardization stack | `fast-mlsirm#1722` exact head `338dbb2d25f32b0e201102e7bf73076846fb57b3`, live stacked base `b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`, four changed files | Verify and integrate the canonical base first, preserve the Rust production owner, then reacquire exact-head recovery and review evidence | **Mergeable against its live stacked base, not direct protected `main`.** No predecessor evidence is approval for this head/base pair. |
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

1. Let the two `.github#1986` dispatch runs reach terminal state and prove
   same-repository status publication; then prove the cross-repository path
   tracked by `.github#1929`.
2. Let `.github#1902` and `.github#1999` acquire exact-head hosted checks
   and independent review; ordinary auto-merge is armed behind protection.
3. Revalidate `fast-mlsirm#1692` on its unchanged head after the owner repair,
   merge ordinarily, and
   produce immutable release evidence before changing the released-version
   claim.
4. Continue the Rust numerical stack in dependency order, starting with each
   PR's live base rather than assuming every Ready PR targets protected `main`.
