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

The Goal remains active. This document is an RCA and execution plan, not a
release-wide license verdict or proof of completed hosted gates.
