# Lead handoff — run_812ab0f36dc8

Exported: 2026-09-19T11:39Z (UTC) / 20:39 KST  
Outgoing lead worktree: `/Users/seonghobae/orca/workspaces/fast-mlsirm/orchestration-lead-fmls-v4`  
Coordinator handle (run): `term_f29498be-5bdd-462c-ae4a-9ed0daf41b83`  
Nested status path that works when `send --run run_812ab0f36dc8` hits `dispatch_run_mismatch`: `--run run_b22de9a1c59d`  
Context pressure at handoff: ~83% — no new implementation this turn.

Companion log: `LEAD_QUEUE.md` (append-only session notes).

---

## Do not touch

| Asset | Detail |
|-------|--------|
| **A4 Oakes fit** | **STOPPED 2026-09-20 by user.** PID 2716563 ABSENT. Stop record `/data/orca/workspaces/a4-oakes-s1-0114/A4_STOP_RECORD_20260920.txt` (+ billing-snapshots copy). Issue #2074 (13 threads/1 running). **Do NOT restart** until immutable release with parallel fix + remeasure pass. |
| Resource-guard shed | WITHDRAWN 2026-09-18. Memory is per-session MCP, not workers. Silent unless new info. No local cargo/maturin on studio; new workers → **MacBookAir**. |

---

## #2021 progress work (keep the product)

The MBA implementation worker **finished** and was **released** just before handoff. Keep the **draft PR / issue work**, not a live agent:

| Field | Value |
|-------|--------|
| Issue | [#2021](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/2021) |
| Draft PR | [#2033](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2033) — `feat(em): opt-in EM progress for fit_two_tier_grm / fit_bifactor_grm` |
| Head | `19b8c34353e766ad9c7859be4f24abbb1e60401a` |
| Branch / worktree (MBA) | `seonghobae/fmls-2021-em-progress` · `…/fmls-2021-em-progress` |
| Dispatch (settled) | `ctx_0c55169ff156` / `task_b1b0d664fee5` — `succeeded` + terminal released |
| Scope shipped | Opt-in per-E-step progress callback; silent default; char test; Bock & Aitkin (1981) locators on issue/PR |
| CI | rollup **PENDING** at handoff (org concurrency) |
| Constraint | Progress is for the **next** long fit — not A4 |

If a new lead needs CI follow-up on #2033, start a **new** MBA worker; do not expect `ctx_0c55169ff156` to still be live.

---

## Active workers

**None.** All recent MBA dispatches settled:

| Dispatch | Goal | Outcome |
|----------|------|---------|
| `ctx_0c55169ff156` | #2021 EM progress | succeeded → #2033 draft; released |
| `ctx_2457db769b2f` | Rebase #2031 | succeeded; released |
| `ctx_e6ddfedae5e9` | #2030 analytic Hessian | succeeded → #2031 draft; released |
| `ctx_cf29cbf4a170` | Semgrep via #2015 | succeeded; released |

Note: `worker-list --include-remote` sometimes shows `host_unavailable` for live MBA workers; trust `worker-show` / `worker-read`. MBA WS `ws://10.6.0.11:6768`.

---

## Priority PRs / CI

### Merge bar (delegated 090948Z)

Admin-bypass merge **only** when: **fail=0 ∧ pending=0** ∧ sole blocker is unobtainable approval.  
Before merge: comment ruleset requirement, absence of approval subject, and numeric check counts.  
Never push with failures or pending. No ruleset edits. No self-APPROVE / undraft-as-owner.

### #2015 — Semgrep red-gate (main unblocker)

| | |
|--|--|
| URL | https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2015 |
| Head | `ced16804` (Semgrep trio + fail-closed allowlist + #2011 Oakes fixture regen) |
| State | non-draft, MERGEABLE, **BLOCKED**, rollup **PENDING** |
| Evidence | Semgrep already passed on prior head; local `bifactor_oakes_mirt` PASS after Oakes regen |
| Blocker | Org Actions concurrency — suites **QUEUED since ~09:16Z**; still pending at handoff |
| Lead actions taken | Cancelled batches of non-priority palette/sentinel/bolt queued runs (35 then 45); comment on PR |

**Next:** poll via **GraphQL** (REST often 403 rate-limited). When fail=0∧pending=0∧approval-only → admin-merge with numeric comment. Landing this clears Semgrep for the rest of the queue (late-life #209 pattern).

### #2031 — analytic M-step Hessian (#2030)

| | |
|--|--|
| URL | https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2031 |
| Head | `4be809bc` (rebased onto main; CHANGELOG-only conflict) |
| State | **draft**, MERGEABLE, BLOCKED, rollup PENDING |
| Do not | undraft/merge until owner call; optional s1 wall recount later |

### #2033 — EM progress (#2021)

Draft; CI PENDING. Keep; do not undraft casually.

### Still drafts (person-side; behind #2015)

| PR | Head | Note |
|----|------|------|
| [#2005](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2005) | `696a2ca1` | Block partial-pattern; measured ~1.2× not ~13× |
| [#2018](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2018) | `21ec51df` | Rayon E-step; s1 floor=0 on **real** `e_step_n_threads` path (bootstrap `n_jobs` is NOT rayon — hardcoded `e_step_n_*=1`) |

### #2011

Open non-draft; historically FAILURE/queued on CodeQL-compat/strix/noema. Contains Semgrep commit already peeled into #2015.

---

## Evidence archives (s1)

| Path | Contents |
|------|----------|
| `/data/orca/workspaces/measure-archive/20260919T111048Z/` | `result-0114.json` (2218.79s), `result-2005.json` (2051.54s), logs, `rayon_repro_out/`, TASK_*.md |
| `/data/orca/workspaces/verify-2003/` | Instrumented #2022 / rayon measure worktree (no live measure PIDs) |

s1 measure/rayon/watcher PIDs: **all dead** at last check. Nothing to stop there.

---

## Standing technical facts (do not re-derive)

1. **M-step dominates** at q=241 (~57%); person-side ceiling ~2.35× (#2022).  
2. **FD sweeps** ~434/M-step; analytic Hessian is the right cut (#2030 → #2031).  
3. **Semgrep gate** scans whole PR-head tree; fix belongs on main (#2015), not only on draft file lists.  
4. **Bootstrap** does not forward `e_step_n_threads` — follow-up, not a floor failure (#2018 lead comment).  
5. Prefer **GraphQL** for GH; REST burns rate limit quickly on cancel/poll loops.

---

## Next actions for incoming lead (ordered)

1. **Poll #2015** until fail=0∧pending=0; then merge under delegated criteria (comment numbers first). Prefer canceling *stuck* in_progress Strix/Noema that pin concurrency over canceling #2015 itself.  
2. **Hold drafts** #2031 / #2033 / #2005 / #2018 — no undraft unless owner. After #2015 lands, Semgrep on dependents should clear without porting.  
3. **#2033 CI** — if stuck queued, same concurrency playbook; optional tiny MBA worker only for CI fixes.  
4. **A4** — report only on **exit or anomaly**; do not re-report wall clock with no progress.  
5. **Do not** start new production CP3; do not local cargo on studio; MBA for new workers.  
6. Status to coordinator: `--type status` via `run_b22de9a1c59d` if `run_812ab0f36dc8` send mismatches ambient dispatch `ctx_58234cf10952`.

---

## SSH Air → Pro (2026-09-19T19:55Z)

| Field | Value |
|---|---|
| Purpose | Two-host probe only. Air calls Pro. Not for A4, not for s1. |
| Client | MacBookAir `seonghobae@10.6.0.11` |
| Server | This Pro `seonghobae@192.168.68.15` (`Seonghoui-MacBookPro-2.local`) |
| authorized_keys | `/Users/seonghobae/.ssh/authorized_keys` on the Pro |
| Air key | `~/.ssh/id_ed25519`, comment `fmls-air-to-pro`, fingerprint `SHA256:TAbdCYhqdaZsIfgnkbGbLGY+oaPp1N0TWJb9ZowTE+4` |
| Host key check | Not disabled. ED25519 `SHA256:mftphosUSku+5JWZkCR1Cs69bolaB5KfEC5HBeq7hiU`, ECDSA `SHA256:DtfVBROGm1IldknFAWGxhjVlYT1zH2RQ7r2X0SlHZiY`, RSA `SHA256:NCp6xeL4HjbwmW+CWXoSIIhigFuOOgPFP/zDCw8fBHo` match `/etc/ssh` and are in Air `known_hosts`. |

## SSH Air → s1 (2026-09-19T21:10Z)

| Field | Value |
|---|---|
| Purpose | #2048 SubprocessExecutor probe only. Not A4, not the Pro, not research numbers. |
| Client | MacBookAir, key `~/.ssh/id_ed25519`, comment `fmls-air-to-pro` |
| Fingerprint | `SHA256:TAbdCYhqdaZsIfgnkbGbLGY+oaPp1N0TWJb9ZowTE+4` (ED25519). Same key as the Air→Pro row. |
| Server | `seongho@192.168.68.3` (`s1.cluster.seonghobae.me`). Not `seonghobae@192.168.68.15`. |
| authorized_keys | `~/.ssh/authorized_keys` for user `seongho`. One `fmls-air-to-pro` line appended. File still has 7 `ssh-` entries; existing lines were not replaced. |
| Host key check | Not disabled. |

## Environments

| Name | Selector | Endpoint |
|------|----------|----------|
| MacBookAir | `--on MacBookAir` / env `87567e14-…` | `ws://10.6.0.11:6768` |
| s1 | env `7b813a1a-…` | `ws://192.168.68.3:8790` — A4 only for heavy fit |

---

## Actions usage — two queries, both kept (2026-09-19T23:15Z)

Do not use the month-specified daily response to deny the unfiltered monthly row. They are different queries. API version on both new reads: `X-Github-Api-Version-Selected: 2022-11-28`. Do not treat `pricePerUnit * quantity` as a substitute for the `netAmount` field. Do not glue a cross-repo sum onto one row.

| Query | File | Bytes | sha256 |
|---|---|---|---|
| `GET /orgs/ContextualWisdomLab/settings/billing/usage` | `billing-snapshots/usage-unfiltered.json` | 10894 | `118b30a2e966e88eb8bab2425eb4466a66c30c4b86c6bfc28a47ff6f3e122f81` |
| `GET .../usage?year=2026` | `billing-snapshots/usage-year-2026.json` | 10896 | `9114c4f88bcfd97083590f454bc4274356e215f0cbe6348558a3adfabbc2cdc4` |
| `GET .../usage?year=2026&month=9` | `billing-snapshots/usage-2026-09.json` | 814901 | `a6e9b945a43730b53ded0e6fc4c9fd9460e03f8bf135a52c3b650271f68bd58f` |

Unfiltered bandscope `Actions Linux` monthly row: `date=2026-09-01T00:00:00Z`, `quantity=1785715`, `netAmount=10714.29`, `discountAmount=0`. The year-only read seconds later is `quantity=1785716`, `netAmount=10714.296`. The earlier citation `1781018` / `10686.108` is a prior meter reading of this monthly-row shape, not a row in these two files, and not disproved by the daily query. fast-mlsirm `Actions Linux ARM` in both monthly reads is one row, `quantity=25`, `netAmount=0`.

The month=9 file has no quantity `1781018` and no single ARM quantity-25 row. bandscope `Actions Linux` there is 19 daily rows, each `netAmount=0`; their quantity sum is 68457. fast-mlsirm ARM there is quantity 8 and 17. Those facts belong only to `year=2026&month=9`.

## Recent status message IDs (nested run)

- `msg_8e2323834f7f` — MBA live despite list host_unavailable; s1 measure archived  
- `msg_6e083ccae59a` — cancelled 45 queued bot runs  
- #2021 worker_done → #2033 (acked `delivery_ab98df9bd3a5`; worker released)

End of handoff.
