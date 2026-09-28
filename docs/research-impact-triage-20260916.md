# Research-impact triage — late-life-anxiety reanalysis (2026-09-16)

Scope: every open issue/PR in `ContextualWisdomLab/fast-mlsirm` that can change an
estimate/SE/score/interval/diagnostic reported by
`ContextualWisdomLab/late-life-anxiety-reanalysis` (Class A), or that can hide such a
defect via test/CI/graph/review tooling (Class B). No fixes implemented here — triage
and routing only.

## Method

- Pulled all open issues (227) and open PRs (97) via paginated GitHub GraphQL, cached
  to local JSON (`issues_pages.json`, `prs_pages.json` in the run's scratch dir).
- Filtered by keyword match against the study's dependency surface (poly GRM,
  bifactor GRM, signed-slope/reflection, DIF purification/FIPC anchoring, EAP/expected
  score, GPU/bootstrap, quadrature, reliability, Graphify, CodeGraph, CI governance),
  narrowing 324 open items to ~30 candidates.
- Confirmed the actual call path with `codegraph_explore` against both repos:
  - `late-life-anxiety-reanalysis` scores almost every subscale through **R `mirt`**
    (fixed-item-parameter FIPC via `mirt()` + `fscores(method="EAP")`), **not**
    fast-mlsirm.
  - The **anxiety-symptoms (Y) scale**, the study's primary DV, is scored through
    `fast_mlsirm.polytomous.score_polytomous` / `polytomous_expected_response`
    (`analysis/generate_expected_scores.py`), which calls Rust `fit_poly_unidim`
    (GRM/GPCM) directly. This is the one place fast-mlsirm code runs the actual
    reported numbers today.
  - The DT (distress-tolerance) bifactor + multigroup + bootstrap work
    (`bifactor_grm.rs`, `poly::fit_poly_multigroup`) is the blocked/in-progress
    replacement for the current 85 CPU-hour bootstrap (#1912); its correctness
    matters once that work lands, and issue #1927 already flags a citation risk in
    that exact code.
  - Every FIPC scoring script (`ac_fipc_*`, `y_anxiety_fipc.R`, `decentering_fipc.R`,
    `dt_bifactor_fipc_*`) fixes **all** anchor items with **no invariance/DIF check**
    — this is the concrete gap #1874/#1891 were filed against.
- Remaining ~190 open issues and ~78 open PRs are bulk-classified **C (unrelated)**
  below by subsystem, without a per-item write-up — they touch model families
  (LSIRM, KSIRT, Mokken, RSM, 2PL, multilevel/cross-classified, LLTM, facets,
  subscores, decision/enterprise, metering/billing, essay scoring, CEFR) that the
  reanalysis never calls, per the call-graph evidence above and AGENTS.md's model
  scope.

## Class A — can change a reported estimate/SE/score/interval/diagnostic

| # | Title | Risk (one line) | Study path hit | Repro |
|---|---|---|---|---|
| **1870** (PR, open, test-only) | test(grm): pin signed-slope recovery for reverse-keyed graded items | Reported live: fast-mlsirm returns exactly `0.000` slopes for reverse-keyed graded items — collapses "inverted" to "uninformative." `fit_grm`/`fit_poly_unidim` share the Rust GRM core that `score_polytomous` calls for the study's Y (anxiety) EAP scores. **No fix PR exists yet — only this regression test.** | Direct (`score_polytomous` → `fit_poly_unidim` → GRM core) | Not checked — test exists (`tests/test_grm_signed_slope_recovery.py`) but requires a from-source Rust build to run in this worktree; not built due to time budget. |
| **1881** (issue) + **1883** (PR, test-only) | fix(mixed): `fit_mixed_items` can't represent a reverse-keyed item; `fit_mmle_2pl` has no orientation rule | Same defect family as #1870 (reflection/sign-convention ambiguity) in the mixed-format estimator. Not on the Y-scale's direct call path today, but shares root cause and is explicitly named in the study's dependency surface. | Indirect (same reflection-canonicalization defect class as #1870) | Not checked (build cost) |
| **1874** (issue) + **1891** (PR) | feat(dif): purification, effect sizes and anchor-eligible sets for polytomous DIF (FIPC precondition) | Filed directly by the reanalysis after an FIPC run fixed **all 14 items as anchors with no invariance test** — every age-group FIPC comparison in the study (AC, Y, decentering, DT) currently has zero DIF screening on its anchor set. Dichotomous purification (`mantel_haenszel_dif_purified`, `logistic_dif_purified`) exists; polytomous purification for GRM anchors does not. | Direct — this is the actual scoring pipeline's anchor-selection gap | Not reproduced (feature gap, not a numeric bug — confirmed by reading the issue's own gap analysis) |
| **1880** (issue) + **1892** (PR, test-only) | fix(dif): `jg_class` applies Jodoin-Gierl bands to the 2-df omnibus and a Nagelkerke R², both wrong targets | `logistic_dif`'s classification (A/B/C) compares the wrong quantity to the wrong boundary; feeds directly into the anchor/DIF screening #1874 is closing the gap for. If FIPC anchor sets end up filtered by `jg_class`, this bug misclassifies which items are safe anchors. **No fix PR — only #1892 pins current (wrong) behavior.** | Direct once #1874's anchor screening lands | Not checked |
| **1912** (issue, in progress) | GPU-parallel polytomous bifactor + multigroup fitting for the 390-replicate joint bootstrap (85 CPU-hours today) | Umbrella for the bootstrap CI the study reports for the DT bifactor score. Stage 1 (bifactor_grm.rs) and stage 2 (multigroup) are merged (`d91cf2c3`, PR #1926). Stage 3 (Oakes SE) and stage 4 (two-tier) have **no open issue/PR yet** — nothing to triage there today, but they are upcoming and on-path. | Direct (future DT bootstrap CIs) | N/A — tracking issue |
| **1921** (PR, open, CONFLICTING) | feat(bifactor): GPU-parallel polytomous bifactor, Lord-Wingersky recursion, and joint bootstrap (stage 5) | Implements the actual GPU bootstrap. Currently `mergeable=CONFLICTING`. Review already flagged it sets crate-wide `erasing_op = "allow"` / `identity_op = "allow"` in `Cargo.toml` to silence 24 clippy findings the PR describes as "test-fixture-only false positives" — that's a blanket gate weakening, not a scoped suppression (see #1905, which documents those 24 findings as genuinely all false positives, so the *classification* is probably right, but the *suppression scope* is wrong: it hides future real `erasing_op` bugs project-wide, including in the bootstrap's own index arithmetic). | Direct | Reproduced (review comment on file, PR state confirmed CONFLICTING via `gh pr view`) |
| **1927** (issue) | docs(bifactor-multigroup): verify Bock & Zimowski (1997) locators against the full text | PR #1926 (merged) cited page/equation locators for multigroup pooling in `bifactor_grm.rs` / `poly::fit_poly_multigroup` from a chapter the maintainer doesn't have full-text access to. If the locators are wrong, the *formula*, not just the citation, may be off — this already-merged code is what stage-2 multigroup pooling runs today. | Direct (merged code, unverified against source) | Not checked — requires library access the maintainer doesn't have |
| **1929** (issue) | fix(quadrature): remove unsourced numeric defaults for quadrature node counts in monotonicity/bifactor APIs | **Maintainer-confirmed defect (2026-09-13 rejection of 41 points).** Project standard is **≥121 Gauss-Hermite nodes per dimension** for study-representative grids (nuisance/specific grids 121/241/481, QMC draws ≥5,000). On `main`, `crates/mlsirm-core/src/quadrature.rs` hard-caps `SUPPORTED_Q = [7,11,15,21,31,41]` — the bifactor GRM cannot request 121 nodes at all — and `BifactorGrmConfig` defaults to `q_general=21`/`q_specific=11`; Python-side monotonicity/bifactor adapters (#1928, `focal_expected_total_score_monotonicity`) default `q_nuisance=41`/`q_specific=41`. Every quadrature-based integral the study reports (EAP scoring, expected-score monotonicity, marginal reliability, the DT bifactor bootstrap) is exposed to this under-integration risk until the cap is lifted and the defaults are sourced or made required. | Direct — quadrature underlies EAP/expected-score/reliability/bootstrap integrals across the whole study | Reproduced — `SUPPORTED_Q` cap and both default sites confirmed to exist on `main` per the issue body; a fix is already in flight as a separate worker (`fix/1929-quadrature-defaults`), not duplicated here. |

### Dependency-aware fix order for Class A

0. **#1929 quadrature cap/defaults** (already claimed by `fix/1929-quadrature-defaults`,
   not re-dispatched here) — lift `SUPPORTED_Q` above 41 and source or require the
   node-count arguments. This is a soft precondition for (2) and #1912 stage 3–5:
   any new bifactor/monotonicity fixture or benchmark written for those should use
   ≥121 nodes/dimension once the cap is gone, and should not be written against the
   current ≤41 cap in the meantime (per maintainer rule, 2026-09-13). Independent of
   #1870/#1880/#1874/#1881/#1921.
1. **#1870 fix** (GRM signed-slope core bug) — blocks nothing else, but is the
   highest-confidence live numeric defect on the study's actual call path. Do first,
   independently.
2. **#1927** (verify bifactor multigroup locators) — independent of everything else;
   run in parallel with (1). Blocks trusting stage-2 output, which stage-3/4/5 build on.
3. **#1880 fix** (`jg_class` targets) — independent of (1)/(2); can run in parallel.
   Should land *before* #1874's anchor-eligible-set feature consumes `jg_class`.
4. **#1874/#1891** (polytomous DIF purification + per-focal-group anchors) — depends
   conceptually on (3) if it wires through `jg_class`; otherwise independent. This is
   the biggest study-validity gap (untested FIPC anchors across every scale).
5. **#1881/#1883 fix** (`fit_mixed_items`/`fit_mmle_2pl` orientation) — same defect
   family as (1); can share the (1) fix's root-cause analysis. Do after (1) since a
   correct GRM reflection rule may inform the mixed-format rule.
6. **#1921 rework** (scope the clippy suppression, resolve merge conflicts) — blocks
   the GPU bootstrap the study needs to replace its 85 CPU-hour CPU run. Should also
   rebase on (0) once merged, since it implements the bifactor bootstrap that (0)'s
   node-count fix directly affects. Independent of (1)–(5) otherwise.

**Parallelizable now:** #1929 (already claimed), #1870, #1927, #1880 (four
independent workstreams; #1921 is a fifth but should sequence after #1929 merges).
**Sequential:** #1874/#1891 should wait on #1880; #1881/#1883 benefits from #1870
landing first but is not strictly blocked; #1921 should rebase on #1929.

## Class B — can hide a Class-A-style defect (tooling)

| # | Title | Risk | Repro |
|---|---|---|---|
| **1907** (issue) | `opencode-review` fail-closed immediately after dispatch, blocking merge on 29 of 43 otherwise-green PRs | Required check fails in 5–8s without actually reviewing; this is currently the single biggest reason Class-A fix PRs (including several above) cannot land even once approved. Fix this or the fix-order above stalls at the merge gate. | Reproduced — issue documents the exact required-check list and failure timing from `gh api`. |
| **1732** (issue) | Test governance: pytest suite can pass end-to-end while individual tests silently `skip`/xfail | 24+ skip sites found, including scientific/GPU/Rust-ownership paths — a GRM or bifactor regression test could report green while actually skipped. | Reproduced (issue cites exact skip-site count against a pinned commit). |
| **1869** (issue) | Test governance: dedicated Rust "Statistical Studies" CI jobs pass when their exact test-name filter matches nothing | Same failure mode as #1732 but on the Rust dedicated-job path — a renamed/removed GRM or bifactor Rust test could silently stop being run while CI stays green. | Reproduced (issue cites the workflow lines). |
| **1837** (issue) | CodeGraph: same-name callee over-approximation for scaling-test module imports | Directly undermines the reliability of the graph tooling this very triage (and this repo's CLAUDE.md-mandated workflow) leans on for "does the study call this" questions. Confirmed to be a real over-approximation in a specific case, scope of the general class unknown. | Reproduced by the reporting PR (#1836 review), per issue text. |
| **1862** (issue) + **1863** (PR, draft) | Graphify: same-name nested-callback binding crosses lexical scope | Graph tooling correctness bug in the extractor the repo's review workflow depends on; a mis-bound callback could hide a real call-path change in a PR review. Fix exists as a draft PR (15 files changed) but isn't merged. | Reproduced (issue gives exact repro files/commit). |
| **1847** (issue) | Graphify: cluster-only shrink guard behavior after duplicate-node normalization needs investigation | Open investigation, not confirmed as a defect yet — the shrink guard *might* be correctly refusing an unsafe overwrite, or might be miscounting nodes. | Not reproduced — issue is explicitly an open investigation, no conclusion drawn yet. |
| **1833** (issue) | Graphify: workspace `Cargo.toml` produces zero nodes and is silently absent from the graph | Low severity (one manifest file, not source code), but is exactly the kind of silent coverage gap that would make a graph-based review miss a change. | Reproduced (warning captured verbatim in the issue). |
| **1905** (issue) | Clippy: 374 warnings, 24 deny-level `erasing_op` confirmed all false-positive (record only, not a fix backlog) | Not itself a defect — it's the classification #1921's blanket suppression should have respected by scoping to just those 24 sites instead of the whole crate. Listed here because #1921 cites this issue's number as justification for a broader suppression than this issue actually supports. | Reproduced (issue is the classification record itself). |

No action needed beyond what's already filed for #1847 (investigation) and #1905
(record); both are already at the right resolution for their risk level.

## Class C — unrelated to the study (bulk, not itemized)

Confirmed unrelated by call-graph (study never imports/calls these) and by
AGENTS.md's declared model scope (MLSIRM/MLS2PLM simple-structure family only):

- **Different model families entirely:** LSIRM/DLSJM (#1828, #1831, #1845, #1849,
  #1850, #1853, #1713 and neighbors), KSIRT, Mokken, RSM, 2PL/MMLE-only paths not
  reached via `fit_poly_unidim`, multilevel/cross-classified/multiple-membership
  (#565 and the `fix(multilevel)` train), LLTM, facets, subscores, decision-engine,
  CDM/DINA, nominal-model-only issues, CEFR proficiency calibration, automated essay
  scoring, "enterprise" verticals (#397, #404, #551, #607–#609, #621, #633).
  Roughly 150 open issues + 60 open PRs fall here — almost entirely a long
  `fix(<module>): seal/bound/preflight/replay ... before ...` defensive-validation
  backlog for modules the reanalysis never touches.
- **General security/release hardening not on the numeric path:** CSP headers
  (#1903), JSON depth-counter floor (#1914), subprocess output/time bounds (#1918),
  non-finite-score JSON parse hooks (#1924), SBOM/provenance (#1691, #1763), release
  packaging (#1597, #1630). These matter for the repo generally but touch reporting/
  tooling code the study doesn't invoke.
- **Docs/ADR housekeeping:** #1436, #1523, #1819 — traceability and numbering
  process fixes, no numeric or tooling-reliability effect.
- **CI hardening PR #1717** ("fix(ci): harden acquisition provenance and GPU
  evidence") explicitly states in its own body that it changes no likelihood/
  estimation/scoring semantics — verified, left as C.
- **Draft PR #1772** ("pin EAP binary64 reduction identity before optimization") is
  a *guard against a future regression*, not a current defect — it's a draft, so the
  guard isn't active yet. Borderline B; not elevated to the table above because
  nothing is currently exploiting the gap it guards against, but flag it to whoever
  next touches EAP/marginal-reliability performance.

## What's left

- Reproduction of #1870, #1881, #1927 requires a from-source Rust build
  (`MATURIN_NO_INSTALL_RUST` path or a working `cargo`/`rustc`), which this triage
  pass didn't do (time budget). The coordinator should build before dispatching the
  #1870 fix, to get a real before/after delta rather than trusting the issue text.
- #1927 needs the actual Bock & Zimowski (1997) chapter (library-access blocker
  noted in the issue itself, not something this triage can resolve).
- Stage 3 (Oakes SE) and stage 4 (two-tier) of #1912 have no open issue/PR yet —
  nothing to route today; re-check when that work starts.
- **#1929** (quadrature node-count cap/defaults) is already claimed by a separate
  worker on `fix/1929-quadrature-defaults`; not re-dispatched by this triage. Per
  maintainer rule (2026-09-13): the project standard is ≥121 Gauss-Hermite nodes per
  dimension for study-representative grids (121/241/481 for nuisance/specific,
  QMC draws ≥5,000), `SUPPORTED_Q` must be raised above the current `41` cap, and no
  fixture/benchmark/equivalence test representing study settings should be written
  against node counts below that standard while this is in flight.
