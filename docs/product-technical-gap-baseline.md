# Product and technical gap baseline

Status: **Non-authoritative point-in-time product-completion inventory**<br>
Observed at: **2026-08-20**<br>
Protected-main basis: **`04d0bc21a2a20693bcf16108cd76d394fe844d23`**<br>
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

- package version **`0.8.0`**;
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

## 7. Representative current pull-request evidence

This table is intentionally **not exhaustive**. It records high-leverage live
work observed on 2026-08-20 against protected
`main@04d0bc21a2a20693bcf16108cd76d394fe844d23`. Every row is
**IMPLEMENTED_ON_ACTIVE_PR / RECHECK_REQUIRED**, never shipped truth.

| PR | Observed head | Observed role | Completion dependency / caution |
| --- | --- | --- | --- |
| [#951](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/951) | `1f9055bbc2e87648194b61aa13248611e54b8167` | configuration integer hardening plus Rust-required automatic backend and buyer/runtime truth | open, non-draft and mergeable at observation; current-head checks/reviews must be re-fetched; overlaps canonical product docs |
| [#1070](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1070) | `ce188a2ab882e333400c6c3018183837799f7ae1` | isolates NumPy parity behind explicit `fit_reference`/CLI reference surfaces | open at the snapshot; re-fetch readiness, reviews, checks, and the overlap with #951/#626 before acting; do not create two backend authorities |
| [#1005](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1005) | `ed9868c1b3636d0470dc82367e80a43036483018` | Rust continuous-time/AR longitudinal Rasch estimator and recovery evidence | open, non-draft and mergeable at observation; exact source/review/check evidence must survive integration |
| [#1014](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1014) | `34697b6df58c1424654bc890571a3dfbe806fd97` | crossed and weighted multiple-membership estimator | GitHub metadata reported non-draft at observation, while the PR body declares it stacked and not independently merge-ready before #1005; re-fetch and preserve both scientific slices explicitly |
| [#1008](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1008) | `42b849c4c05c4464205c3afe71e39ce8c0cdae2a` | relation-aware structural model-selection governor | open, non-draft and mergeable at observation; depends on accepted likelihood/recovery/scoreability evidence rather than policy alone |
| [#1003](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1003) | `27a831865c2f1ee42710dceb27a2e074c13ad254` | governed item-bank lifecycle JSON/HTML reports and replay hardening | open, non-draft and mergeable at observation; report integrity does not itself complete calibration/linking/exposure/drift evidence |
| [#1012](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1012) | `64e6dad6a62b2b391c0632c7a2a93cc2fdce0ca8` | durable 500-rep GRM recovery evidence workflow | open, non-draft and mergeable at observation; workflow evidence must remain bound to the unchanged scientific source and exact head |
| [#1071](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1071) | `8b5bb0351dae5db97b666cc6c0b2423b79209f76` | bounded statistical-study deadline increase | open, non-draft and mergeable at observation; longer deadlines must still produce terminal, retained, fail-closed evidence rather than hiding nonconvergence |

Additional open work can change this dependency graph at any time. A completion
or merge decision must begin with a fresh repository-wide PR and writer sweep.

## 8. Product and technical gap matrix

| Gap ID | Priority | Required outcome | Existing issue / PR evidence | Completion test |
| --- | --- | --- | --- | --- |
| GAP-01 | P0 | Freeze a bounded 1.0 capability, support and maturity matrix; do not equate planned research with GA | [#621](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/621), [#636](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/636), [#648](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/648) | every public capability is classified, supported versions match metadata, and the release gate makes no valuation/certification claim |
| GAP-02 | P0 | One ordinary Rust/PyO3 numerical owner; NumPy only on explicit reference/parity surfaces | [#626](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/626), [#627](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/627), PRs [#951](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/951) and [#1070](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1070) | production config/API cannot silently select Python numerics; missing/incompatible Rust fails before result-affecting work |
| GAP-03 | P0 | Complete non-atomistic multilevel, cross-classified, multiple-membership and longitudinal estimation with identification and recovery | [#565](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/565), PRs [#1005](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1005) and [#1014](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1014) | realistic aligned bias/MAE/RMSE/coverage/convergence and temporal leakage tests pass; both stacked scientific deltas survive |
| GAP-04 | P0 | Relation-safe factor retention, structural model selection and identified exploratory multidimensional estimation | [#608](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/608), [#633](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/633), [#551](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/551), PR [#1008](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1008) | no winner is forced without relation-appropriate tests, held-out evidence, scoreability and true-structure recovery |
| GAP-05 | P1 | Close rubric, generated-item, scoring, RAG, essay, enterprise-issue and item-bank lifecycles without parallel contracts | [#397](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/397), [#404](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/404), [#607](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/607), [#609](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/609), PR [#1003](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1003) | one immutable assessment/rubric/scoring lineage reaches pilot, calibration, validation, lifecycle and report evidence without provider coupling or silent state promotion |
| GAP-06 | P0 | Independently test equations and fitted estimands against explicitly matched mature engines | [#1077](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1077) | versioned capability×engine matrix, fixed-parameter equation conformance first, aligned fitted-result comparisons, visible disagreement register |
| GAP-07 | P0 for validated claims | Add preregistered external validity, language/site/time transportability, fairness and criterion evidence profiles | [#1078](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/1078) | external evidence is genuinely held out; claim register narrows automatically on absent, failed or indeterminate evidence |
| GAP-08 | P0 | Stabilize public artifact, schema, serialization, fingerprint, capability and migration contracts | [#637](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/637), [#653](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/653), [#499](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/499) | strict RFC 8259 artifacts, no environment-dependent capability downgrade, versioned loaders/migrations, cross-language canonical fixtures |
| GAP-09 | P0 | Complete release/support/supply-chain evidence and truthful compatibility policy | [#648](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/648), [#623](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/623), [#636](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/636), documentation audit PARTIAL states | supported line/runtime/platforms are tested; wheel, SBOM, provenance, license, rollback and vulnerability process are source-hash-bound |
| GAP-10 | P1 | Publish capacity/performance envelopes instead of isolated speed claims | [#403](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/403), [#563](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/563) | representative N×item×dimension×facet×time workloads report latency, throughput, peak RSS/VRAM, failure ceilings and CPU/GPU parity |
| GAP-11 | P0 operations | Eliminate orphaned workflow identities and retain complete terminal statistical/release evidence | [#809](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/809), PR [#1071](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1071) | complete paginated workflow registry is reconciled; supported workflows remain; statistical studies terminate with durable evidence |
| GAP-12 | P1 product | Prove one buyer-visible vertical through a downstream host while preserving repository ownership | [#397](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/397), [#404](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/404), [#607](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/607), [#584](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/584) | source evidence → governed observations → Rust calibration → uncertainty/fairness/validation → accessible report → downstream human decision is replayable end to end |

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
2. Complete #626/#627 Rust ownership and fail-closed evidence.
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

This issue closes the self-consistency gap by requiring explicit
parameterization mappings, neutral fixed-parameter fixtures, aligned
fitted-result comparisons, a capability×engine matrix, license isolation and a
visible disagreement register. Mature external implementations are validation
instruments only and never become production/build/package dependencies.

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
- which repository owns each remaining concern; and
- the next root-cause-changing action without relying on chat history.
