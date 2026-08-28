# Product and technical gap baseline

Status: **Non-authoritative point-in-time product-completion inventory**<br>
Observed at: **2026-08-28T04:41:11Z**<br>
Protected-main basis: **`main@45627700…`**<br>
Repository: **`ContextualWisdomLab/fast-mlsirm`**

## 1. Purpose and authority

This document answers one bounded question:

> What remains before `fast-mlsirm` can make a defensible technical-GA claim, and what additional evidence remains before a downstream product can make a validated domain or high-stakes use claim?

This file is an inventory and routing aid. It is **not** a competing PRD, TRD,
architecture, ADR, release manifest, or statement of shipped capability.
Canonical authority remains:

- [`docs/PRD.md`](PRD.md);
- [`docs/TRD.md`](TRD.md);
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md);
- [`docs/documentation_coverage.md`](documentation_coverage.md);
- the status-bearing ADR graph; and
- [issue #621](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/621), or its explicitly accepted successor, for cross-cutting documentation fitness.

Protected `main` is shipped truth. An issue, open pull request, green check on an
unmerged head, review comment, branch description, scheduler state, or this
inventory is evidence only. Before acting on any row below, re-fetch:

1. protected-main SHA;
2. pull-request head and live base;
3. draft/ready and mergeability state;
4. current-head reviews and unresolved threads;
5. required checks; and
6. active writer/overlapping-path ownership.

No predecessor-head check or review transfers after a head or live-base change.

## 2. Executive disposition

At the observed protected-main SHA, `fast-mlsirm` is a substantial Rust/PyO3
psychometric measurement core, but a general technical-GA or universal
high-stakes readiness claim is not yet defensible.

The strongest remaining completion dependencies are:

1. finish one Rust-owned ordinary production numerical boundary with no silent
   Python fallback;
2. integrate and validate the multilevel, multiple-membership, longitudinal,
   model-selection, and recovery slices that are currently split across issues
   and active PRs;
3. add independent cross-engine equation and fitted-result conformance evidence;
4. add preregistered external-validity and transportability evidence profiles;
5. freeze a bounded 1.0 capability/support matrix instead of treating every
   research or planned model as part of GA;
6. complete stable artifact/version/migration, release, support, supply-chain,
   benchmark, and rollback evidence; and
7. prove at least one buyer-visible end-to-end workflow through an owning
   downstream product without moving hosted identity, consent, persistence, or
   decision governance into this reusable core.

A package can reach **technical GA** while particular domain or high-stakes use
profiles remain unvalidated. Technical correctness, construct validity,
transportability, fairness, and decision utility are separate claims.

## 3. Current protected-main product truth

The observed protected main declares:

- package version **`0.9.1`**;
- Python **`>=3.12`**;
- Maturin/PyO3 bindings to the Rust workspace;
- PyPI classifier **`Development Status :: 3 - Alpha`**; and
- an “early high-performance toolkit” product description.

Protected main already provides substantial evidence and usable primitives,
including:

- Rust/PyO3 likelihood, optimization, diagnostic, scoring, linking, CAT/ATA,
  and selected GPU/CPU parity paths;
- deterministic simulation and true-parameter recovery infrastructure;
- governed rubric, scoring, evidence, RAG, essay, enterprise-issue, and
  item-bank contracts;
- fail-closed validation and bounded-resource controls;
- package/wheel/reinstall, fuzz, security, SAST, and protected-check gates;
- accessible standalone reports and content-addressed provenance patterns; and
- canonical PRD/TRD/architecture, V&V, threat-model, standards-watch,
  UML/ERD, and traceability families.

The protected-main documentation audit still classifies several release-critical
families as **PARTIAL**, notably:

- public interface/version/serialization/fingerprint contracts;
- reusable-core operability and recovery;
- security/data-governance navigation;
- release/migration/rollback/provenance/licensing navigation;
- requirements traceability and selected UML/ERD coverage; and
- root README/AGENTS/CLAUDE/Architecture/PRD/TRD/CHANGELOG alignment.

Those states are not cosmetic documentation tasks. They identify product
contracts that a buyer, downstream integrator, or maintainer still cannot
reconstruct reliably without source archaeology.

## 4. Product boundary

### 4.1 `fast-mlsirm` owns

- domain-neutral psychometric numerical kernels;
- public simulation, fitting, scoring, diagnostics, comparison, linking, CAT,
  ATA, recovery, and evidence contracts that are explicitly integrated on
  protected main;
- Rust-first numerical ownership, deterministic Python validation/marshalling,
  bounded resource controls, and versioned reusable artifacts;
- package-level V&V, benchmark, security, interoperability, provenance, and
  release evidence; and
- source-text-free reports and handoff contracts.

### 4.2 Downstream products own

`ContextualWisdomLab/psychometrics-commons` or another explicitly owning host
owns, as applicable:

- tenants, accounts, OIDC/SSO/SCIM and authorization;
- participants, sessions, consent, data-rights and purpose limitation;
- hosted persistence, object storage, queues, APIs, UI and billing;
- operational item banks and restricted test content;
- human review, approval, administration and incident workflows;
- domain-specific external validation data and high-stakes decision policy; and
- regulated deployment, retention, deletion and audit execution.

The downstream host may consume `fast-mlsirm` only through a traceable,
versioned handoff: a released package and schema version, a versioned API/schema,
or an immutable content-addressed artifact reference. The consumer records the
package/artifact version, source commit, schema version, and environment
provenance used for each result. A floating branch checkout or unrecorded
implementation import is not a reusable integration contract. `fast-mlsirm`
must not depend on that host to remain installable and useful as a standalone
library.

### 4.3 Explicit non-goals for this repository

- a universal validity or fairness certification;
- a hosted assessment/session database;
- direct storage of operational PII or restricted test content;
- automatic causal claims from observational scores;
- provider-specific LLM execution inside the numerical core;
- treating one external package as an unquestionable oracle;
- a machine-generated acquisition valuation or guaranteed sale price; and
- declaring every planned model family part of a 1.0 support promise.

### 4.4 Versioned downstream handoff

The reusable-core boundary is actionable only when a consumer can identify the
artifact it is allowed to import and the owner of the surrounding lifecycle.
The current handoff therefore follows these repository contracts:

- [`docs/scoring_assessment_contracts.md`](scoring_assessment_contracts.md) and
  [`docs/scoring_execution_contracts.md`](scoring_execution_contracts.md) define
  the package-owned request, observation, scoring, and execution surfaces;
- [`docs/enterprise_issue_evidence_contracts.md`](enterprise_issue_evidence_contracts.md)
  defines source-free evidence handoff for an owning product; and
- [`docs/adr/0001-domain-neutral-measurement-boundary.md`](adr/0001-domain-neutral-measurement-boundary.md),
  [`docs/adr/0003-content-addressed-measurement-contracts.md`](adr/0003-content-addressed-measurement-contracts.md),
  and [`docs/adr/0013-continuous-execution-and-documentation-governance.md`](adr/0013-continuous-execution-and-documentation-governance.md)
  define ownership, immutable provenance, and documentation authority.

Consumers must pin a released package/artifact schema and record its source and
environment provenance. A downstream host owns participant/session/consent,
authorization, persistence, raw content, human decisions, and regulated
retention; this baseline does not create a second database or HTTP contract.
The handoff is therefore reusable across `psychometrics-commons` and other
consumers while `fast-mlsirm` remains independently installable.

## 5. Completion profiles

### 5.1 Technical alpha

This is the current declared package line. Useful APIs may exist, but public
contracts, support scope, scientific evidence, compatibility, and operational
surfaces can still change before 1.0.

### 5.2 Technical GA — reusable measurement core

A technical-GA profile requires a bounded, versioned list of supported public
capabilities. For every listed capability, the profile must provide:

- one ordinary Rust/PyO3 production numerical owner;
- fail-closed behavior when that owner is missing or incompatible;
- explicit identification, estimand, model/estimator compatibility, resource,
  missingness, and convergence contracts;
- true-parameter recovery or inferential error evidence appropriate to the
  claim, including Monte Carlo uncertainty where stochastic;
- independent cross-engine conformance where a scientifically equivalent
  implementation exists;
- stable public API and artifact schemas with migration/rollback policy;
- exact supported Python/platform/backend matrix;
- 100% repository-required production statement/branch coverage and public
  docstring evidence;
- benchmark/capacity evidence and bounded failure behavior;
- security, fuzz, package/reinstall, SBOM, provenance and licensing evidence;
- current support and vulnerability-reporting policy; and
- one unchanged exact head satisfying all required reviews and checks.

A capability that lacks the required evidence remains experimental, research,
planned, or explicitly outside the GA profile; it does not block unrelated,
bounded GA capabilities.

### 5.3 Validated domain profile

A domain profile binds the technical core to one assessment, rubric/item-bank,
population, setting, language, time period, criterion, and intended score use.
It additionally requires content/response-process, internal-structure,
external-variable, transportability, fairness, and consequence evidence.

A domain profile is versioned independently of the Python package. A package
upgrade does not automatically validate an old profile, and a validated profile
does not approve every other use of the same estimator.

### 5.4 High-stakes use profile

A high-stakes profile additionally requires the owning product’s legal,
privacy, security, human-governance, accessibility, adverse-impact,
monitoring, incident, appeal, and decision-policy controls. This status cannot
be inferred from software tests, parameter recovery, cross-engine agreement,
or a passed package release gate.

## 6. Status vocabulary used here

This baseline reuses the repository’s canonical capability vocabulary:

- **IMPLEMENTED_ON_PROTECTED_MAIN**;
- **IMPLEMENTED_ON_ACTIVE_PR**;
- **PARTIAL**;
- **ACCEPTED_ARCHITECTURE**;
- **PLANNED**;
- **RESEARCH_ONLY**;
- **DOWNSTREAM**;
- **SUPERSEDED**;
- **REJECTED**; and
- **OUT_OF_SCOPE**.

For live PR rows, **RECHECK_REQUIRED** is only a snapshot annotation. It is not a
new canonical capability-maturity state.

## 7. Current pull-request evidence

The following table records high-leverage live work observed on
2026-08-28T04:41:11Z against protected `main@45627700…`. Every row is
**IMPLEMENTED_ON_ACTIVE_PR / RECHECK_REQUIRED**, never shipped truth. A green
check on any row is not a protected-main capability until the PR is merged.

| PR | Observed head | Observed role | Completion dependency / caution |
| --- | --- | --- | --- |
| [#1525](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1525) | `5558a46a…` | validates the Rust `subscore_analysis` result envelope before NumPy marshalling (issue [#1524](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1524)) | ready; ordinary required CI, Rust, package, fuzz, coverage, and security checks passed, but required `opencode-review` failed and no independent review is present; active-PR evidence is not protected-main capability |
| [#1523](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1523) | `b76092d4…` | aligns the traceability matrix with the protected-main v1/v2 lineage-anchor status and keeps unavailable v2 producer/estimator work out of implemented claims | ready; required CI, Rust, package, fuzz, coverage, and security checks passed; independent approval remains required |
| [#1522](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1522) | `b73d7686…` | seals multigroup/multilevel population-label admission against caller coercion callbacks (issue [#1521](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1521)) | ready; required CI, Rust, package, fuzz, coverage, and security checks passed; independent approval and protected merge are absent |
| [#1519](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1519) | `c9d39672…` | refreshes this buyer-facing protected-main product/technical gap inventory and corrects the longitudinal citation/status record | ordinary required CI, Rust, package, fuzz, coverage, and security checks passed; required `opencode-review` failed; independent approval and protected merge are absent |
| [#1518](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1518) | `35ffcdc1…` | Rust ordered posterior-draw summary with weighted uncertainty, credible-level ambiguity, and fail-closed evidence admission (issue [#1484](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1484)) | ready; required CI, Rust, package, fuzz, coverage, and security checks passed; independent approval and protected merge are absent |
| [#1516](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1516) | `19ecd219…` | seals Rasch CML structural response traversal and native result replay before public marshalling (issues [#1515](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1515)/[#1517](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1517)) | ready; required checks are rerunning on this exact head after a native-count boundary repair; an earlier non-required Strix run reported a report-mapping failure, and independent approval remains required |
| [#1510](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1510) | `a4e03857…` | bounds KSIRT response evidence and replays native result shape before NumPy marshalling (issues [#1507](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1507)–[#1514](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1514)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1506](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1506) | `21babc2f…` | bounds Mokken zero-cell structural traversal before NumPy materialization (issue [#1505](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1505)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1504](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1504) | `e8de9421…` | applies the cross-engine report accessibility palette changes | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1501](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1501) | `34338763…` | bounds EBDIF item evidence before allocation (issue [#1500](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1500)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1494](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1494) | `fb3f7e24…` | seals LLTM scientific evidence admission before NumPy protocols (issues [#1493](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1493)–[#1499](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1499)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1492](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1492) | `9feae4de…` | preserves integer safety across essay contracts reload (issue [#1491](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1491)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1481](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1481) | `7946f4b8…` | bounds parallel-analysis observed data before dense marshalling (issues [#1480](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1480)–[#1489](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1489)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1479](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1479) | `057a4082…` | replays factor-retention result invariants (issue [#1478](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1478)) | ready; current CI, Rust, package, fuzz, coverage, and security checks passed; independent approval still required |
| [#1471](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1471) | `8ce2ae2e…` | cuts the 0.9.2 release | ready; current checks passed, but the release parent remains review-blocked; stack children must be synchronized after any parent change |
| [#1476](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1476) | `f3c66f0e…` | replays item-bank evidence-reference identity (issue [#1475](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1475)) | targets release PR #1471; no required checks were reported on this child branch, so it is not protected-delivery evidence |
| [#1473](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1473) | `bf1a7782…` | binds validation profiles to preregistration time (issue [#1472](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1472)) | targets release PR #1471; no required checks were reported on this child branch, so it is not protected-delivery evidence |
| [#1417](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1417) | `05e06766…` | extends the Rust residual interaction-map result envelope (issue [#1412](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1412)) | draft; current checks passed, but review/merge is absent and child PRs depend on this parent |
| [#1436](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1436) | `a90db19f…` | documents the Rust polytomous period artifact | draft child of #1417; clean/mergeable only describes the stack candidate, not protected main |
| [#1457](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1457) | `94699c5b…` | exposes the Rust interaction-map explained-share projection | draft child of #1417; currently conflicting and requires parent synchronization before review |

At this observation, GitHub REST enumerated **20 open pull requests** and GitHub
search enumerated **51 open issues**. The current direct-to-main queue is
dominated by bounded native-boundary and result-replay fixes; the release stack
is rooted at #1471, and the interaction-map stack is rooted at #1417. The
long-lived product gaps remain #621 (bounded 1.0 capability/support matrix) and
#565 (multilevel/multiple-membership/longitudinal completion). The Rust-owned
ordinary-production boundary has materially advanced through the integrated
backend/runtime work; issues #626 and #627 are closed, but the remaining public
capability and release evidence still prevents a universal technical-GA claim.

This list is a reproducible snapshot, not a merge instruction. A completion or
merge decision must begin with a fresh repository-wide PR and writer sweep,
including exact head/base, dependency stack, reviews, unresolved threads,
required Checks, and active path ownership.

The previous six-row snapshot is obsolete: those listed PRs are no longer open,
and the current queue above was re-fetched from the live repository. Their
integrated capability evidence remains part of protected main where the merge
was completed; the active rows above remain candidates until their exact heads
pass current reviews and protected merge gates.

### 7.1 Superseded lineage record

Two orphaned Sentinel security branches were deleted on 2026-08-25 because
protected main already ships strictly stronger fixes for their scopes:

- `sentinel-fix-json-recursion-conformance-4916450064032858492` (JSON recursion
  DoS in `cross_engine_conformance.py`) — superseded by merged
  [#1330](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1330)
  (`a25833a0`: raw JSON depth guard + `tests/test_cross_engine_conformance_json_depth.py`).
- `sentinel-medium-fix-unbounded-json-loading-11914195049005804093` (unbounded
  JSON loading in ops scripts) — superseded by main's
  `scripts/_bounded_json.parse_json_bounded(..., max_bytes=...)` hardening in
  `build_pr_queue_governance.py`, which bounds GitHub stdout bytes beyond the
  branch's proposal.

Issues [#1300](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1300),
[#1301](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1301), and
[#1303](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1303) were
closed on 2026-08-25 with protected-main evidence: `_TRUSTED_REAL_CONTROL_TYPES`
excludes Boolean identity (`mhrm.py` lines 46–51) with regression
`test_mhrm_real_control_boolean_admission.py`; `classify_model_relation()`
enforces exact-type admission and replays `__post_init__` invariants
(`model_relation.py` lines 144–146).

## 8. Product and technical gap matrix

| Gap ID | Priority | Required outcome | Existing issue / PR evidence | Completion test |
| --- | --- | --- | --- | --- |
| GAP-01 | P0 | Freeze a bounded 1.0 capability, support and maturity matrix; do not equate planned research with GA | [#621](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/621), [#636](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/636), [#648](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/648) | every public capability is classified, supported versions match metadata, and the release gate makes no valuation/certification claim |
| GAP-02 | P0 | One ordinary Rust/PyO3 numerical owner; NumPy only on explicit reference/parity surfaces | closed issues [#626](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/626) and [#627](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/627); the automatic-backend and reference-isolation slices landed on protected main via merged [#951](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/951)/[#1070](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1070) | production config/API cannot silently select Python numerics; missing/incompatible Rust fails before result-affecting work |
| GAP-03 | P0 | Complete non-atomistic multilevel, cross-classified, multiple-membership and longitudinal estimation with identification and recovery | [#565](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/565); crossed multiple-membership estimator landed via merged [#1014](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1014), and continuous-time/AR longitudinal Rasch landed via merged [#1005](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1005) | realistic aligned bias/MAE/RMSE/coverage/convergence and temporal leakage tests pass; the remaining profile must preserve its declared estimand and recovery evidence |
| GAP-04 | P0 | Relation-safe factor retention, structural model selection and identified exploratory multidimensional estimation | [#608](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/608), [#633](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/633), [#551](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/551), PR [#1008](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1008) | no winner is forced without relation-appropriate tests, held-out evidence, scoreability and true-structure recovery |
| GAP-05 | P1 | Close rubric, generated-item, scoring, RAG, essay, enterprise-issue and item-bank lifecycles without parallel contracts | [#397](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/397), [#404](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/404), [#607](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/607), [#609](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/609), PR [#1003](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1003) | one immutable assessment/rubric/scoring lineage reaches pilot, calibration, validation, lifecycle and report evidence without provider coupling or silent state promotion |
| GAP-06 | P0 | Independently test equations and fitted estimands against explicitly matched mature engines | [#1077](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1077) closed as COMPLETED after the reusable conformance provenance manifest landed ([#1082](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1082)); residual validation-family execution evidence tracks under [#1092](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1092)/[#1094](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1094)/[#1096](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1096)/[#1152](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1152) | versioned capability×engine matrix, fixed-parameter equation conformance first, aligned fitted-result comparisons, visible disagreement register |
| GAP-07 | P0 for validated claims | Add preregistered external validity, language/site/time transportability, fairness and criterion evidence profiles | [#1078](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1078) | external evidence is genuinely held out; claim register narrows automatically on absent, failed or indeterminate evidence |
| GAP-08 | P0 | Stabilize public artifact, schema, serialization, fingerprint, capability and migration contracts | [#637](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/637), [#653](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/653), [#499](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/499) | strict RFC 8259 artifacts, no environment-dependent capability downgrade, versioned loaders/migrations, cross-language canonical fixtures |
| GAP-09 | P0 | Complete release/support/supply-chain evidence and truthful compatibility policy | [#648](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/648), [#623](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/623), [#636](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/636), documentation audit PARTIAL states | supported line/runtime/platforms are tested; wheel, SBOM, provenance, license, rollback and vulnerability process are source-hash-bound |
| GAP-10 | P1 | Publish capacity/performance envelopes instead of isolated speed claims | [#403](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/403), [#563](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/563) | representative N×item×dimension×facet×time workloads report latency, throughput, peak RSS/VRAM, failure ceilings and CPU/GPU parity |
| GAP-11 | P0 operations | Eliminate orphaned workflow identities and retain complete terminal statistical/release evidence | [#809](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/809), PR [#1071](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1071) | complete paginated workflow registry is reconciled; supported workflows remain; statistical studies terminate with durable evidence |
| GAP-12 | P1 product | Prove one buyer-visible vertical through a downstream host while preserving repository ownership | [#397](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/397), [#404](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/404), [#607](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/607), [#584](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/584) | source evidence → governed observations → Rust calibration → uncertainty/fairness/validation → accessible report → downstream human decision is replayable end to end |
| GAP-13 | P1 downstream UI | When a hosted consumer has a web surface, make UI states and interactions auditable rather than treating a static screenshot as product evidence | [`docs/adr/0016-figma-buyer-evidence-design-boundary.md`](adr/0016-figma-buyer-evidence-design-boundary.md) (merged via [#1130](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1130)), [`docs/figma_product_design_packet.md`](figma_product_design_packet.md), Storybook interaction-testing guidance | ADR records the exact Figma file ID (`qD34PfMH8Kr41tFdqLCkem`); a Storybook inventory covers the ten UI/UX dimensions below; each high-risk story has an event-driven interaction assertion and an accessibility result |

## 9. Ordered completion sequence

### Stage 0 — establish live ownership and exact evidence

1. Re-fetch all open PRs, bases, heads, reviews, threads, checks and overlapping
   paths.
2. Preserve unique scientific deltas; close or supersede duplicates only with an
   explicit lineage record.
3. Do not widen a PR merely because another lane is waiting on Actions or review.
4. Resolve infrastructure failures at their root without weakening scientific,
   security, coverage or independent-review gates.

### Stage 1 — close the technical-GA numerical boundary

1. Reconcile #951 and #1070 so one public backend/reference contract survives.
2. Retain the protected-main closure evidence for #626/#627 Rust ownership and
   fail-closed behavior in the support matrix.
3. Define the first bounded 1.0 capability/support matrix under #621/#648.
4. Reject advertised-but-unimplemented model×estimator combinations before
   fitting and remove normal-path `NotImplementedError` surfaces from the GA
   profile.

### Stage 2 — integrate scientific foundation and recovery

1. Land longitudinal and multiple-membership work in dependency order while
   preserving both exact scientific slices.
2. Integrate factor-retention/model-selection policy only with the required
   relation, likelihood, scoreability, held-out and recovery evidence.
3. Complete durable exhaustive recovery studies with MCSE/intervals and explicit
   failed-replication classes.
4. Add exploratory multidimensional loading estimation only after its
   identification and rotation contracts are accepted.

### Stage 3 — independent numerical validation

Implement #1077 in bounded slices:

1. capability and estimand inventory;
2. parameter-mapping schemas and neutral equation fixtures;
3. fixed-parameter equation conformance;
4. fitted-result alignment and comparisons;
5. scheduled/release evidence, disagreement register and accessible reports.

External engines remain isolated test instruments, not runtime or package
dependencies.

### Stage 4 — external validity and transportability

Implement #1078 through one reusable validation-profile contract, then apply it
to a license-compliant synthetic/open/de-identified portfolio. Keep technical,
construct, transportability, fairness and decision-utility evidence separate.
A failed profile narrows the corresponding claim rather than failing unrelated
technical capabilities.

### Stage 5 — one closed buyer workflow

Choose one initial vertical—automated essay scoring, reference-free RAG
measurement, or enterprise issue measurement—and prove the complete handoff
through the owning downstream product. The first accepted vertical must include:

- exact assessment/rubric/item/task/rater/model/source/version provenance;
- fallible human/automated rater calibration;
- recovery, scoreability, DIF/invariance and held-out validation;
- source-free accessible JSON/HTML with exact-value tables;
- human review/decision boundaries; and
- no claim that correlation, schema validity, model fit or one judge equals
  construct validity.

### Stage 6 — artifact, release and support hardening

1. Freeze versioned public API/artifact schemas and explicit migrations.
2. Prove old supported serving/results artifacts load or fail with a documented,
   stable migration status.
3. Run clean-install, upgrade, rollback and wheel-reinstall rehearsals.
4. Emit signed source/build provenance, SBOM, checksums, license/NOTICE and
   reproducibility manifests.
5. Publish current support/security policy and capacity envelope.
6. Release only from an unchanged exact head with every required check and
   review terminal-success.

## 10. Buyer-visible acceptance gates

### 10.1 Numerical and scientific

- no silent Python production fallback;
- no model-name-only relation or compatibility inference;
- explicit identification and failure classification;
- realistic true-parameter recovery with bias, MAE/RMSE, coverage, convergence
  and Monte Carlo uncertainty;
- CPU single-thread/multithread determinism and real GPU parity where enabled;
- independent cross-engine conformance or an explicit justified
  `not_comparable` state;
- external/transportability evidence before making corresponding domain claims;
- no high-stakes claim from correlation, fit, schema conformance or recovery
  alone.

### 10.2 API, artifact and interoperability

- semantic versioning and a bounded deprecation policy;
- canonical schema/version/fingerprint preimages and cross-language fixtures;
- strict RFC 8259 JSON with no NaN or infinity extension tokens;
- content-addressed immutable scientific and validation artifacts;
- explicit capability profiles and no environment-dependent partial bundles;
- backward-compatibility, migration, rollback and rejection tests; and
- source-text-free reusable numerical artifacts.

### 10.3 Quality and security

- production statement coverage 100%;
- production branch coverage 100%;
- public Rust/Python API docstrings 100%;
- property, metamorphic, fuzz, hostile-input and denial-of-service tests;
- exact runtime/platform/backend support matrix;
- dependency, OSV, SAST, CodeQL, Trivy, Scorecard, Strix, package and fuzz gates;
- no secret, PII, restricted test content or provider response in release or
  billing telemetry; and
- current threat model, responsible disclosure and support policy.

### 10.4 Release and supply chain

- reproducible source/dependency/environment manifests;
- SPDX SBOM using a stable published specification;
- SLSA-compatible build provenance with pinned immutable actions/tools;
- source, wheel, report, model and validation artifact hashes;
- clean build/install/reinstall/upgrade/rollback rehearsal;
- license and redistribution review for datasets, external engines and models;
- release notes generated from authoritative fragments; and
- no draft standard or future revision represented as current certification.

### 10.5 Buyer workflow and accessibility

- one complete downstream workflow is replayable from evidence to result and
  human decision;
- every number in charts is also available in an exact-value table;
- keyboard, screen-reader, no-JavaScript and print/PDF evidence where applicable;
- missing, abstained, failed, excluded, not-applicable and indeterminate remain
  distinct; and
- reports expose limitations and next actions, not only a score or badge.

### 10.6 UI/UX, Figma, and Storybook boundary

The protected `fast-mlsirm` package has no web frontend or Storybook workspace;
it is a reusable numerical/core-contract library. A downstream product that
adds a web surface must own its UI implementation, design tokens, Storybook
inventory, and Figma file. This repository must not acquire a UI dependency or
pretend that a screenshot proves an interaction contract.

The existing buyer-review design packet records Figma file ID
`qD34PfMH8Kr41tFdqLCkem` in
[`docs/figma_product_design_packet.md`](figma_product_design_packet.md). The ADR
binding that identity is protected-main truth as
[`docs/adr/0016-figma-buyer-evidence-design-boundary.md`](adr/0016-figma-buyer-evidence-design-boundary.md),
merged through [#1130](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1130).
A future Figma-backed change must keep the file ID in its ADR and keep
Code Connect disabled unless a separate architecture decision authorizes it.

For a downstream Storybook, each story is both a named visual state and a
replayable test case. The story starts from explicit props/context, its
`play` function emits a realistic user event, and assertions inspect the
observable result (role, accessible name, text, focus, callback, URL, or
machine-readable value). Required scene/edge coverage is:

| UI/UX dimension | Required scene and event evidence |
| --- | --- |
| Accessibility | keyboard-only focus order, accessible names/roles, screen-reader state, contrast, reduced-motion, and an automated WCAG 2.2 audit |
| Touch & Interaction | pointer, touch, keyboard, disabled, loading, cancellation, double-submit, and focus-restoration events |
| Performance | empty, representative, and upper-bound datasets with render/interaction budgets and no unbounded DOM growth |
| Style Selection | design-token default, dark/high-contrast, error/success, and token-regression snapshots |
| Layout & Responsive | narrow/wide viewport, zoom, long labels, overflow, RTL, and orientation changes |
| Typography & Color | long/localized text, font fallback, contrast, color-independent status, and numeric formatting |
| Animation | entrance, interruption, timeout, reduced-motion, and state-change completion without hiding content |
| Forms & Feedback | blank, invalid, server error, retry, async pending, success, and keyboard submit flows |
| Navigation Patterns | deep link, back/forward, unsaved state, modal escape, route failure, and restored focus/scroll |
| Charts & Data | no data, one point, dense data, outlier, tooltip keyboard access, exact-value table, and export/error states |

This inventory is a downstream acceptance contract, not a claim that this
library currently ships a UI. Storybook's official interaction-testing model
uses stories plus `play` functions to simulate clicks, typing and submission
and assert the result; the corresponding evidence is linked in
[`docs/doctoring/ui-ux-storybook-evidence.md`](doctoring/ui-ux-storybook-evidence.md).

## 11. Claim register

| Claim | Minimum evidence | Current baseline disposition | Family scope / claim limitations |
| --- | --- | --- | --- |
| “The package implements the declared equation” | Rust unit/property tests plus #1077 fixed-parameter cross-engine/neutral-fixture conformance where comparable | PARTIAL | Declared model paths only; independent engine agreement is still incomplete. |
| “The estimator recovers parameters” | ADEMP simulation, alignment, bias/MAE/RMSE/coverage/convergence/MCSE | PARTIAL | Evidence exists for selected estimator families, not every advertised family or data regime. |
| “CPU and GPU are equivalent” | real non-skipped GPU execution against CPU `f64` reference under declared tolerances | PARTIAL | Only kernels with a real GPU execution and an explicit CPU reference are covered. |
| “This score measures the intended construct” | content, response-process, internal-structure and external-variable evidence for a named profile | OUT_OF_SCOPE | Requires a named downstream domain profile; it is not a universal package claim. |
| “The interpretation transports” | #1078 held-out site/language/time/rater/revision evidence | PLANNED | Transportability must be shown for the declared held-out units and time window. |
| “The use is fair” | lawful subgroup support, DIF/invariance, threshold/error and consequence evidence | PLANNED | Evidence is profile-specific and must include the supported subgroups and decision context. |
| “The product improves decisions” | preregistered policy/utility evaluation against baselines; causal language only with identified design | DOWNSTREAM | The owning host controls the policy, outcome, intervention and decision-utility evidence. |
| “The package is technical GA” | bounded support matrix plus all technical-GA gates in this document | PLANNED | The current package line is technical alpha until every declared GA gate is evidenced. |
| “The product is approved for high-stakes use” | validated profile plus downstream legal/privacy/security/human-governance controls | OUT_OF_SCOPE | High-stakes approval belongs to a validated downstream profile and its owning governance process. |

## 12. Issues created from this review

### [#1077 — independent cross-engine numerical conformance](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1077)

This issue defined the self-consistency gap by requiring explicit
parameterization mappings, neutral fixed-parameter fixtures, aligned
fitted-result comparisons, a capability×engine matrix, license isolation and a
visible disagreement register. It closed as **COMPLETED** on 2026-08-24 after
the reusable source-free conformance provenance manifest landed
([#1082](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1082));
execution-side validation evidence continues under the open validation family
([#1092](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1092),
[#1094](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1094),
[#1096](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1096),
[#1152](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1152)).
Mature external implementations are validation instruments only and never become production/build/package dependencies.

### [#1078 — external validity and transportability profiles](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1078)

This issue defines preregistered, purpose-bounded validation profiles that keep
technical, construct, transportability, fairness and decision-utility evidence
separate. It requires held-out site/language/time/rater/revision units,
criterion-quality limitations, explicit failed/indeterminate states and no raw
PII or restricted content in reusable artifacts.

## 13. Documentation and PR maintenance rule

This baseline should be refreshed only when a material product-completion
boundary changes. It must not become a manually maintained mirror of every
open PR.

A refresh shall:

1. pin the observed protected-main SHA and date;
2. query live PR/issue state rather than copying prior snapshots;
3. preserve the canonical maturity vocabulary;
4. classify active work as active only;
5. update links and gap ownership without rewriting canonical PRD/TRD/ADR
   authority;
6. remove rows that are integrated, superseded or rejected; and
7. route any changed protected-main maturity to #621 or its accepted successor.

The preferred long-term form is a generated/read-only view whose durable inputs
are the canonical documentation graph, protected-main capability registry, live
GitHub metadata, release evidence and validation manifests.

## 14. Standards and research status

Use published standards as normative references and drafts/revision projects as
watch items only.

- The 2014 *Standards for Educational and Psychological Testing* is the current
  published testing-standard baseline for validity, fairness and score-use
  claims (American Educational Research Association et al., 2014). AERA, APA
  and NCME revision work is a watch item until a new edition is published.
- ISO/IEC 25010:2023 is the current published product-quality model baseline for
  software product quality characteristics and quality evaluation (International
  Organization for Standardization & International Electrotechnical Commission,
  2023).
- The ITC 2018 test-adaptation guidelines govern translation/adaptation and
  cross-language equivalence evidence; translation alone is not validation
  (International Test Commission, 2018).
- RFC 8259 governs strict JSON interoperability and its grammar/encoding
  boundary (The Internet Engineering Task Force, 2017).
- Semantic Versioning 2.0.0 is the public versioning baseline unless a more
  specific package contract is accepted (Preston-Werner, 2013).
- NIST SP 800-218 SSDF 1.1 is the current final SSDF baseline; SSDF 1.2 remains
  a draft watch item until finalized; the SSDF supplies secure-development
  practices rather than a certification (National Institute of Standards and
  Technology, 2022).
- SLSA 1.2 and SPDX 3.0.1 are stable published supply-chain/provenance and SBOM
  baselines; SLSA addresses build provenance and SPDX addresses machine-readable
  licensing/component interchange. Draft successors must not be presented as
  current conformance (Software Package Data Exchange, 2024; Supply-chain
  Levels for Software Artifacts, 2025).

No standard reference in this file is a certification claim.

Research traceability is maintained in the canonical
[`docs/traceability/research-basis.md`](traceability/research-basis.md) index
and the linked primary-source records under [`docs/papers/`](papers/README.md).
The references in this baseline explain the product decision boundary; they do
not replace the model-specific paper-first record required before changing a
formula, estimator, fit statistic, or interpretation-facing output.

The package-literature entries below are included as implementation context, not
as substitutes for primary methodological validation: Chalmers (2012) describes
multidimensional IRT software and its estimation surface; Mair and Hatzinger
(2007) documents extended Rasch model tooling; Rizopoulos (2006) documents
latent-variable and IRT analysis tooling; Robitzsch et al. (2025) documents the
TAM test-analysis modules. Morris et al. (2019) provides the simulation-study
design rationale used by the recovery evidence requirement. Each source is
linked in the APA list below so a reviewer can reconstruct the decision without
access to chat history.

## 15. APA 7th reference baseline

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association. https://www.testingstandards.net/open-access-files.html

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
https://doi.org/10.18637/jss.v048.i06

International Organization for Standardization & International Electrotechnical
Commission. (2023). *Systems and software engineering—Systems and software
quality requirements and evaluation (SQuaRE)—Product quality model*
(ISO/IEC 25010:2023). https://www.iso.org/standard/78176.html

International Test Commission. (2018). ITC guidelines for translating and
adapting tests (Second edition). *International Journal of Testing, 18*(2),
101–134. https://doi.org/10.1080/15305058.2017.1398166

Mair, P., & Hatzinger, R. (2007). Extended Rasch modeling: The eRm package for
the application of IRT models in R. *Journal of Statistical Software, 20*(9),
1–20. https://doi.org/10.18637/jss.v020.i09

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies
to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102.
https://doi.org/10.1002/sim.8086

National Institute of Standards and Technology. (2022). *Secure software
development framework (SSDF) version 1.1: Recommendations for mitigating the
risk of software vulnerabilities* (NIST SP 800-218).
https://doi.org/10.6028/NIST.SP.800-218

Preston-Werner, T. (2013). *Semantic Versioning 2.0.0*.
https://semver.org/spec/v2.0.0.html

Rizopoulos, D. (2006). ltm: An R package for latent variable modeling and item
response analysis. *Journal of Statistical Software, 17*(5), 1–25.
https://doi.org/10.18637/jss.v017.i05

Robitzsch, A., Kiefer, T., & Wu, M. (2025). *TAM: Test analysis modules*
(R package version 4.4-2). https://doi.org/10.32614/CRAN.package.TAM

Software Package Data Exchange. (2024). *SPDX specification 3.0.1*.
https://spdx.github.io/spdx-spec/v3.0.1/

Supply-chain Levels for Software Artifacts. (2025). *SLSA specification 1.2*.
https://slsa.dev/spec/v1.2/

The Internet Engineering Task Force. (2017). *The JavaScript Object Notation
(JSON) data interchange format* (RFC 8259).
https://www.rfc-editor.org/rfc/rfc8259

### Gap: Event Lineage channel weights still lack estimable independent outcomes

LineageWeave ADR 0208 routes channel-weight arithmetic here, while TEPP PR #237
publishes the accepted `tepp.lineage_criterion_anchor.v1` run-level decision.
That artifact does not contain pair-level independent criterion observations.
The legacy Python path's score-floor dichotomization and internally anchored
MLS2PLM therefore cannot be ported and presented as calibrated measurement.

This change adds the Rust continuous-evidence and exact anchor-identity
prerequisite, with 100% line/function/branch coverage for its module. The
estimation result remains explicitly unavailable. Completion requires a TEPP
successor binding independent criterion posterior/outcomes to pair identities,
followed by an accepted estimator ADR, Rust CPU/GPU same-objective path,
true-parameter and known-weight recovery, uncertainty coverage, and protected
integration. Period-report calibration/aggregates remain a separate owner debt
and are not silently bundled into this contract.

## 16. Change boundary for this baseline

This document introduces no production code, numerical formula, public API,
dependency, workflow, database, package version, support promise, release,
certification or changelog entry. It records a point-in-time product-completion
analysis and routes work to existing or newly created issues.

The document is complete when reviewers can determine:

- what protected main actually ships;
- what active PRs may add but do not yet ship;
- which evidence blocks technical GA;
- which evidence blocks domain/high-stakes claims;
- what repository owns each remaining concern; and
- the next root-cause-changing action without relying on chat history.
