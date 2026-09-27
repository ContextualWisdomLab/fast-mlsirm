# CI recovery and release evidence plan

## Objective

Complete the 0.11.5 published 12-wheel license evidence. Restore executable
current-head CI, integrate validated notice/evidence changes, retain unproven
license HOLDs, and issue a final verdict only for a verified zero-HOLD scope.

## Live findings, 2026-09-27 UTC

Five organization runners were online. The control group contains
`cwlab-s1-01` and `cwlab-s1-05`; CodeQL and OpenCode have separate groups.
Group restrictions admit selected central workflow identities and preserve
the boundary between trusted control work and arbitrary PR builds.

Central main `fd2a03e35c454b85108628f1d975ff714417e6d8` already includes
runner-routing changes, including scheduler routing from #2417. No duplicate
selector patch was needed. The original allocation record uses a constrained
linear relaxation and integer allocation; changing host capacity or adding an
optimizer without new service measurements is not justified by this audit.

Both license PRs were drafts, causing repository CI to skip. After rechecking
their exact heads and finding no review threads, they were marked ready:

| PR | Exact head | Local focused checks | New repository CI run |
| --- | --- | --- | --- |
| #2174 | `14887ecd1042664dc75676389663780c173a3e8e` | 10 passed | `36319558931` |
| #2175 | `3d6ffe191289dbc8b376911216e9bdd45c7436ac` | 253 passed, expected duplicate-ZIP warning | `36319571082` |

Their earlier Actions CodeQL jobs succeeded. The new Python/Rust/package jobs
were queued at inspection; skipped old jobs are not evidence of passing CI.

### Notice CI integration

At 12:43:21 UTC, #2174 was merged into its existing stacked base
`codex/regression-hc3-producer-gap-20260925`, merge commit
`9e01372f9ff525205ef453c2bd930a1e634ae049`. This is not adoption on main or
publication. Immediately before the user-authorized admin merge, head remained
`14887ecd1042664dc75676389663780c173a3e8e`, base remained
`58b7b23f7a154c8391c130ebc3aab2dde50bb84b`, the PR was mergeable, and review
threads were empty. CodeRabbit and Devin statuses and Actions CodeQL were
successful; new general CI jobs remained queued. Ten focused tests,
`actionlint` on the publication workflow, `bash -n` on the selector, and
whitespace checks of changed code passed. The change selects reviewed target
notices, rejects unsupported targets, and verifies built-wheel notice bytes.
It changes no model implementation or dependency manifest. Hosted full-suite
success and a release license gate are not claimed by this integration.

A targeted dry-run scheduler dispatch for #2174 created run `36319361497`,
job `108620080034`, on the central main above. It disables PR mutations,
review dispatch, branch updates and merges. Its requested labels are
`self-hosted`, `linux`, `x64`; it was still queued and had no assigned runner.

Control VM `cwlab-s1-05` journal independently confirmed successful cleanup
and changed-scope jobs between 12:35 and 12:36 UTC. At 12:37:51 it started
`scan-pr-queue`. Its worker context binds that execution to earlier run
`36315580579` and the central scheduler at `refs/heads/main`, not to this
canary. Thus control admission works for at least that earlier main run, and
the backlog is draining; this does not prove the new canary has executed.

A preceding failed bootstrap on the same runner was independently bound to
`ContextualWisdomLab/orca`, run `36316726464`, job `108612723362`. Its failed
step is `Enforce Cloudflare Pingora edge policy`, after successful source
materialization. That failure is distinct from runner admission and must not
be treated as a queue flake or bypassed as part of this recovery.

## Next actions and completion evidence

1. Follow the existing canary job until terminal; record its actual runner,
   group, steps and conclusion. Do not restart it after an observation timeout.
2. Audit the new #2174/#2175 runs at their exact current heads. Fix actual
   failures from logs; retain queued checks as pending. Before integration,
   recheck bases, review threads and required gates. The maintainer authorized
   bypass for CI capacity stalls after local validation; that does not justify
   treating a real policy or security failure as passing.
3. Integrate the notice selector and reviewed evidence without importing
   unrelated stacked model changes. Revalidate any changed integration source
   and retain its immutable source archive and lockfile hashes.
4. Continue the [primary license review](nonlinux-upstream-20260926/REVIEW.md):
   eleven binding-union Cargo HOLDs and three Python HOLDs remain. Missing
   grants or native provenance require new bound evidence, not weaker matching.
5. Verify Windows/macOS wheel notices and source bytes, then obtain the actual
   published 12-wheel matrix. Match published hashes to exact source and
   target-specific inventories. Local Linux candidates and source graphs
   remain narrower evidence and cannot substitute for this matrix.

## Follow-up at 2026-09-27 12:52 UTC

Evidence PR #2175 was merged into its reviewed-texts base at
`0dc785b1623eb0493ddd55ea8a892ee357acf56a`, from exact head
`a240f26233f4093d68fac14ef09b555363528a35`. All 253 focused tests and all
29 evidence checksums passed; both hosted review contexts succeeded and no
review threads remained. Source-byte whitespace is deliberately retained.
The Actions CodeQL job remained queued; authorized bypass is not its success.
This stacked merge does not establish main adoption or publication.

The fixed canary job 108620080034 remained queued. Independent guest logs
show other scheduler jobs completing successfully on cwlab-s1-05. Naruon
run 36315355984/job 108613158572 occupied that runner for about five minutes;
its terminal error was a pending dispatch verdict, not a reported vulnerability.
The old compatibility workflow enumerated the entire dispatch run history.
Current central main `f6a50f6a` already bounds that lookup by the required
run's creation time and dispatch event, and adds Actions read permission.
Do not duplicate that existing fix or treat old queued workflow snapshots as
proof that the current fix failed. Follow the canary and verify fresh-head
lookup timing before changing allocation again.

## Additional capacity observed at 2026-09-27 12:54 UTC

The organization now exposes six online runners. Newly registered runner
`cwlab-s2-01` (ID 1065083) has the control label and is already in group 6,
alongside cwlab-s1-01 and cwlab-s1-05. It changed from idle to busy during
observation. The scheduler workflow's exact main ref remains admitted by
the group's restricted workflow list; no group broadening is needed.
Canary job 108620080034 is still live queued without assigned runner.

Guest lifecycle records also show contextual-orchestrator run 36315207621
completing its Actions compatibility shard successfully in about ten seconds
(12:52:32–12:52:42 UTC). Its recorded workflow SHA is
`7dbd1e5a1d976cc117756b70849b8534334eb827`, so this is operational capacity
evidence, not a measurement of central #2433's new source. Keep that
distinction when assessing the fresh lookup fix.

## Verified admission and hourly follow-through

Canary run 36319361497/job 108620080034 reached cwlab-s1-01 in
CWL central control and completed at 2026-09-27T13:08:11Z. Its failure
log states that targeted PR2174 was closed, matching its completed merge.
This proves runner admission but not a successful scheduler verdict.

The maintainer explicitly requested hourly follow-through until merge.
Enabled Orca automation `7c92d308-5b01-4788-8dca-9172a1a66e2d` runs at
each hour in Asia/Seoul, starting 2026-09-27 23:00 KST, in the original
license workspace with session reuse. It carries the PR stack, exact-head
validation, authorized bypass, RCA and license/artifact preservation rules.
The Goal was resumed by the user and is active.

After the original canary became terminal, a new dry-run scheduler dispatch
was accepted for the still-open PR2157 on main. It disables review triggers,
auto-merge, branch updates and merge actions. Follow its new run handle;
this is a current-source acceptance check, not a retry of the closed target.

## License stack integration at 2026-09-27 13:21 UTC

PR2170 merged from exact head `0dc785b1623eb0493ddd55ea8a892ee357acf56a`
into the Windows-notice base at `dd1ca669969c57984fc48f9c69ad63b2abda9a5f`.
Before the authorized bypass, the base was
`163ffae3885ee205b58b2e7cb4f108771638377d`; the PR was mergeable,
both review contexts succeeded, and no review threads remained. Its complete
verifier diff was inspected; the tested verifier, template matcher, fixture
and two test files were byte-identical to the integrated evidence checkout.
The fresh focused run passed 253 tests. Hosted CI remained queued.

PR2154 now has that merge as its head, against base
`e38182f5e296dc46730a04088166214333b6421b`; it remains open and mergeable.
Audit that new identity before the next stack merge. No main adoption or
release acceptance is asserted by this stacked merge.

The new canary is run 36321971364/job 108627427635, central source
`eb59914a0b8bf37b2abfb4ac0c083d1043a263dc`, targeting the open PR2157.
It was confirmed queued after creation. Follow this exact handle.

## Subsequent stack merges at 2026-09-27 13:24 UTC

PR2154 merged exact head `dd1ca669969c57984fc48f9c69ad63b2abda9a5f`
into the soname-binding base at `66d2fd3efbbb3067066f130d26eaf370b3908c51`.
PR2152 then merged that exact head into the primary evidence branch at
`186da53c9b87c111bcb80dbeb2bdaf082430707c`. Each PR was mergeable,
had no unresolved review threads, and had successful CodeRabbit and Devin
contexts before its authorized bypass. Hosted test jobs remained queued.
The Windows pattern conversion and soname/copyleft-label diffs were reviewed;
both resulting verifier/test/fixture trees equal the integrated source that
passed all 253 focused tests. Canonical archive paths remain strict and
runtime-exception copyleft findings stay HOLD.

Next integrate primary evidence PR2139 into main after auditing its current
base/head and complete diff. These stacked merges do not by themselves prove
main adoption, successful hosted gates or release acceptance.

The Goal remains active. This document is an RCA and execution plan, not a
release-wide license verdict or proof of completed hosted gates.
