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

## Primary evidence integrated into main

The PR2139 merge audit found stale NumPy 2.5.3 locks while inspected main
`270865294873c7b0ebcc8ff0659db2d6d8c93a48` and the 0.11.5 artifact evidence
use 2.5.2. Semantic inspection found only NumPy changed in uv.lock and only
the NumPy block changed in each hashed requirements file. Commit
`d810939c5fb5667efb1b43e654e9824afda2398e` restored those three files from
the exact inspected main, without updating unrelated packages. The offline
lock check and all 253 license tests passed in an isolated checkout.

PR2139 was marked ready and merged at 2026-09-27T13:28:11Z into main as
`c8eae1f4e31b357329a1ef6d4c68dcb81cc0d343`. Exact head/base and resolved
review threads were checked immediately before the authorized bypass;
CodeRabbit and Devin contexts succeeded, hosted scan jobs remained queued.
Remote main ancestry and verifier/template/fixture byte equality were
verified after merge. The reviewed license stack is now adopted in main.
This does not clear the remaining HOLDs or prove release artifact acceptance.

## Standalone fuzz lock repair adopted

PR2173 merged exact head `841ec40fb31c49832427d47df1797adf851dd3c9` into
main at `6cb7a2caad1c014f5933e303e9c8fddf1b8cd079`, 2026-09-27T13:35:55Z.
Its sole diff against main was fuzz/Cargo.lock. In a source archive of that
exact head, cargo metadata --locked resolved 151 packages without altering
the lock (SHA256 `4289848b06599ae426ff09e4e5449d3b013a0579f198b89b1f0a436729462def`).
The first offline attempt lacked cached arbitrary1.4.2; fetching the existing
locked dependencies resolved that environmental limitation. No compile was
claimed from this metadata check. Hosted address fuzzing, final CodeQL and
security scans had succeeded, and review threads were empty.

The Python aggregate failure log explicitly reported a cancelled matrix;
the historical compatibility CodeQL failure reported a pending dispatch
verdict, rather than a vulnerability. The draft flag was removed before the
authorized exact-head bypass. Remote main ancestry and equality of the fuzz
lock blob were verified after merge. Keep fresh hosted jobs separate from
these historical receipts and retain all remaining license/artifact HOLDs.

The Goal remains active. This document is an RCA and execution plan, not a
release-wide license verdict or proof of completed hosted gates.

## Central Rust source candidate RCA

Central release collector at exact parent `37a178bf05d0fcfc8557b46699f27d35eab6cae5`
selected `objc2-foundation/src/copying.rs` by conventional filename prefix.
All archive capture/rebinding callers share `archive_license_evidence`; the
fix excludes `.rs` only from implicit discovery and preserves explicitly
declared Cargo/Python license paths. Regression reproduced before repair;
131 focused gate/full-text tests pass after repair. Central PR
https://github.com/ContextualWisdomLab/.github/pull/2446 is stacked on #2347
at commit `eb297a5b` in isolated worktree
`/tmp/fmls-central-license-fix-20260927`. This removes a false candidate,
not missing grants, Apple SDK applicability questions, or release HOLDs.

REST Actions job lookup hit the current API rate limit; preserve canary
`36321971364` rather than dispatching a duplicate. Hourly automation
`7c92d308-5b01-4788-8dca-9172a1a66e2d` remains enabled.

Central PR2446 merged at `09f3ba1dbad4c4d92209b1c0bedd71db16142d6f`
into the #2347 branch at 2026-09-27T14:00:05Z (not main). Exact reviewed
head `baba52d8914779502511915b1da01f8457d58650` integrated current base
`b994c9cb6d3a9e96b3c1ee4246a1171c46499be8`; 131 focused tests passed,
CodeRabbit/Devin succeeded, review threads were empty, and hosted jobs were
queued. User-authorized exact-head admin bypass used. Retain the full
central release stack and downstream immutable-helper adoption as next work.

## Current-main integration and effective helper adoption

Central #2347 conflicts with main `23f36cd56fbe245a06e7a9727cb28d9511154645`
only in two concurrent documentation additions. Isolated ordinary merge
`c9ca98d5e95cbaa2d0e8052c98b2ea7e8055a0be` preserves both additions and
imports current CodeQL/Noema/Strix controls. Source workspace
`/tmp/fmls-central-license-fix-20260927` is clean. Full pytest is live under
exec session `99090`, PID12873, log `/tmp/fmls-central-main-integration-tests.log`;
do not duplicate or treat observation timeout as a terminal result.

Effective helper RCA: release workflow still pinned d67a7175 and thus did not
use merged #2446. Guard test separately retained obsolete aea63e11; its RED
result was 1 failed/2 passed. Commit a3ce4427 in
`/tmp/fmls-central-helper-adoption-20260927` updates all three fixed checkout
pins and guards to immutable c9ca98d5, scripts tree
`cb887976a40493f0e2a945ff1d0f21519f4b5634`, unchanged Strix lock blob.
57 identity/workflow tests pass; all three shipped guards pass against the
actual clean Git checkout with an unrelated caller SHA. Actionlint with
shellcheck disabled and own diff check pass. Proposed stacked PR:
https://github.com/ContextualWisdomLab/.github/pull/2449 .

Global ad-hoc Ruff E9/F/I reports 250 existing dynamic-export/import issues;
no broad lint success claimed. The current-main imported noema transport file
also has an existing trailing blank line; no unrelated rewrite was made.
Latest six runners are online/busy, including s2. Canary36321971364 remains
authoritatively queued; keep that handle. Release grant/artifact HOLDs unchanged.

## Terminal full-suite RCA and corrected executable helper

Integrated-parent test session99090 is terminal: 4,936 passed, 8 failed,
4 skipped, 40 subtests in604.79s. Exact log
`/tmp/fmls-central-main-integration-tests.log`. Do not repeat that handle.
Besides the already-repaired pin test, two production defects caused failures:
install_is_authorized passed an extra GateError argument, and gate merged
license_texts before validating its mapping shape. Commit
`64bb4e7d32667980223c70afda81ffac97209630` removes the erroneous arguments
and reuses the existing mapping validator. Two stale expected-code lists now
retain both disagreement and missing-grant diagnoses. 143 focused tests pass.

PR2449 exact current head `7bc87db7b8e783de481cdc478eaa089e33aa566d` pins
all three checkouts/guards to corrected64bb4e7d, scripts tree
`b15b746d2a1c38501c2ecd6f6c5962c27c9d11af`. 57 pin/workflow tests pass,
three actual Git guards pass; actionlint and own diff check pass. Latest-head
full suite is LIVE exec19087 PID62837, log
`/tmp/fmls-central-2449-exact-tests.log` in helper-adoption worktree.
This replaces the terminal predecessor; preserve the live process and inspect
its exact result before merging2449/2347.

A bounded live job audit showed the three old Noema runs were on GitHub hosted
runners, not self-hosted control occupants. s1-05 journal shows control jobs
complete and advance; historical compatibility jobs still consume five minutes.
Current-main CodeQL controls are retained by this integration. Canary unchanged.

## PR2449 verified merge and remaining root review findings

Exact-head full suite at7bc87db7 is terminal GREEN: 4,944 passed,4 skipped,
40 subtests in438.72s. PR2449 merged into2347 at
`d11c302d91739990eb77d3af3a958720fd7eb950`,2026-09-27T14:27:39Z.
Exact head/base, empty review threads and successful CodeRabbit/Devin were
revalidated; hosted scans remained queued. Authorized bypass used. Remote
ancestry and helper/workflow byte equality verified. Project item is Done.
This is central release-branch integration, not main or release acceptance.

Root PR2347 still has six unresolved threads. Current code confirms direct
tomllib import despite Python3.10 support, four install-hook alias bypasses
(all returned [] for inert Python subprocess/urllib/os aliases and Rust
std::process alias), stale lock-directive documentation, and potentially
colliding artifact namespaces. The install-order test already passes capture
root; inspect the Noema content-review finding before changing that owner.
Next: fix valid shared collector/security/compatibility findings in isolated
stacked work; retain security gates and license refusals.

The existing fast2135 CodeQL proof run36235138881 is terminal failure at
1efddbc9. Connector job evidence shows language detection and central scan
dispatch succeeded; compatibility enforcement jobs108401985319/331 failed.
No actual vulnerability inference follows from this status. Its unresolved
review requires producer evidence; do not resolve from dispatch success.
Latest main de71b9ee includes canonical Strix capacity continuation2448; its
merge-tree with candidate is clean. Do not duplicate that owner repair.

## Shared hook inspection and Python3.10 review repair

Central proposed PR https://github.com/ContextualWisdomLab/.github/pull/2450
(branch codex/release-gate-review-repairs-20260927, helper-adoption worktree)
repairs two verified2347 findings. Eight real-gate alias cases reproduced RED.
The shared scanner now uses stdlib AST for import aliases and dangerous
references, retains all previous patterns, accumulates alias bindings rather
than overwriting them, and recognizes Rust process/network namespaces.
Benign os.path use remains admitted. This is static capability inspection,
not a proof of complete behavioral sandboxing.

Strix fixture now includes captured hook source text, not only filenames;
same-named body changes invalidate the existing fixture digest. No dependency
hook is executed. Python3.10 fallback reuses already-declared tomli; complete
module import and TOML parsing passed with actual tomli under simulated absent
tomllib, not an actual Python3.10 interpreter claim.

Focused gate/capture/whole-text/fanout/collector suite236 passes;57 workflow
contracts and all3 actual Git guards pass. compileall, production-module
Ruff E9/F, actionlint(shellcheck disabled), own diff check pass. Helper source
`5a29e0a5f487799c68f788c42cdc243df27315b8`, scripts tree
`88f38292ed543660de94eee44a110fe47a9a5974`. Exact PRhead
`55de728e8704b4217300c5580cd8cf7e9ddb5076`; full tests LIVE exec1628,
log `/tmp/fmls-root-review-repairs-full-tests.log`. Preserve this process.
Original2450 reviewer-thread adoption happens only after verified merge;
artifact namespace, obsolete directive docs, and Noema content-reader finding
remain separate root2347 work. License and release artifact HOLDs unchanged.
