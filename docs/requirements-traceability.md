# Conversation / Research Requirements Traceability

**Status:** Authoritative coverage matrix for fast-mlsirm  
**Last reviewed against protected main:** 2026-08-09  
**Protected-main reference at baseline cut:** `4d910ed650f384ff882c8b5fba6a8b08fd532236`

## 1. Sufficiency finding

Before this documentation baseline, the repository had strong method-specific
scientific doctoring, release/readiness documents, `AGENTS.md`, `CLAUDE.md` and
feature RFCs, but the conversation-wide architecture was **not sufficiently
consolidated**. In particular there was no root `ARCHITECTURE.md`, no authoritative
component PRD/TRD, no ADR corpus, no canonical logical ERD, no consolidated
UML/C4-style diagram set, and no repository-wide requirements traceability
matrix.

This baseline closes that structural documentation gap. It does **not** claim that
all product/scientific work is complete. Rows below explicitly distinguish
implemented, active-PR, planned and downstream requirements so documentation
cannot make unmerged work appear shipped.

## 2. Traceability matrix

| Requirement family | PRD | TRD | ADR / architecture | Protected-main implementation/evidence | Status / next gap |
|---|---|---|---|---|---|
| Reusable component vs hosted product | PR-001, PR-013 | TR-001 | ADR-0001, `ARCHITECTURE.md` | `AGENTS.md`, `CLAUDE.md`, independent wheel/package/release paths | **Implemented/documented.** Keep Psychometrics Commons product DB/HTTP/session concerns downstream. |
| Rust numerical source of truth | PR-002 | TR-006–TR-009 | ADR-0002 | `crates/mlsirm-core`, PyO3 binding, Rust-default backend assertion, parity/GPU/release tests | **Implemented.** Continue migrating new mathematical features through common binding architecture. |
| True-parameter recovery over correlation | PR-004 | TR-025–TR-027 | ADR-0008 | simulation/recovery diagnostics, statistical-study/release evidence, Rust/Python parity | **Implemented/evolving.** Expand model-specific bias/RMSE/coverage and false-selection recovery as new estimators land. |
| Relation-safe model selection | PR-005 | TR-011–TR-012 | ADR-0004 | `python/fast_mlsirm/model_comparison.py` fail-closed statuses; Rust-backed Vuong selection quantities | **Partially implemented.** Formal distinguishability/score-information support remains an explicit scientific gap before returning non-nested winners. |
| Bifactor scoreability separate from model selection | PR-006 | TR-013 | ADR-0004 | public bifactor scoreability exports on protected main; Rust/Python validation | **Implemented baseline.** Continue categorical reliability/recovery only under explicitly validated definitions. |
| Adaptive factor rotation | PR-007 | TR-014 | ADR-0007 | public `RotationSolution` / criterion exports on protected main | **Implemented baseline/evolving.** Continue broader criterion families and GPU batched starts only with recovery/parity. |
| Canonical Assessment/Rubric/Scoring contracts | PR-003 | TR-003–TR-005 | ADR-0003 | shared versioned contracts; essay/enterprise adapters; package exports | **Implemented.** Prevent downstream duplicate schemas. |
| Rubric → blueprint → generated item | PR-003, PR-011 | TR-019–TR-021 | ADR-0003, ADR-0009 | rubric-centered blueprint/generation contracts documented and exposed | **Partially implemented.** Full semantic screening, artificial-crowd orchestration and governed bank remain convergence work. |
| Governed item-bank lifecycle | PR-011 | TR-021 | ADR-0009, logical ERD, state UML | CAT/ATA/linking/DIF primitives exist; no complete canonical bank lifecycle release yet | **Planned convergence / high buyer-value gap.** |
| Automated essay/general scoring validation | PR-008 | TR-016–TR-018 | ADR-0005 | shared scoring contracts, essay adapters/reports/validation; active PR #579 adds paired range-use evidence | **Implemented core/evolving.** Generalized rater discrimination/range/drift models need separate identification/recovery. |
| Reference-free RAG / LLM judge measurement | PR-008 | TR-016–TR-020 | ADR-0005, ADR-0009 | domain-neutral scoring/rubric infrastructure can represent judge observations; no RAG-specific hidden dependency | **Architecture accepted; integration consumer-specific.** Add canonical adapter only if reusable across independent RAG consumers. |
| Human/LLM rater as fallible measurement instrument | PR-008 | TR-016–TR-018 | ADR-0005 | many-facet/scoring-validation evidence surfaces; exact engine/request provenance | **Implemented baseline/evolving.** Correlation remains descriptive, not validity proof. |
| Multilevel / cross-classified / multiple membership | PR-009 | TR-015 | ADR-0006 | protected main has existing group/cluster diagnostics; PR #566 carries richer reusable contextual/longitudinal contracts | **Active PR / not yet fully integrated.** Numerical estimators require Rust recovery before product claims. |
| Temporal / longitudinal modeling | PR-009 | TR-015 | ADR-0006 | occasion/drift concepts in scoring and active PR #566; TEPP is broader downstream owner | **Active/planned.** Do not call discrete occasion-step AR(1) continuous-time without a distinct model/recovery contract. |
| Low-context-switch CPU multithreading | PR-002 | TR-008 | ADR-0002 | Rust parallel kernels/worker evidence where implemented | **Implemented per-kernel/evolving.** Benchmark actual thread/BLAS configuration for performance claims. |
| GPU acceleration | PR-002 | TR-009 | ADR-0002 | Rust device path + GPU parity/no-skip CI for supported workloads | **Implemented for supported kernels, not universal.** Add kernels only with parity/recovery. |
| Public PyO3/API composition | PR-002, PR-013 | TR-002, TR-028–TR-029 | ADR-0010 | package root currently exposes core + bifactor + rotation families | **Architecture accepted.** Consolidate legacy/secondary binding registration into one maintainable registry as features converge. |
| Untrusted JSON/provider trust boundary | PR-003, PR-010 | TR-004–TR-005, TR-022 | ADR-0003 | bounded scoring/rubric/provider contracts and adversarial tests | **Implemented/evolving.** Semantic validity remains separate from parse validity. |
| Privacy without utility-destroying blanket masking | PR-010 | TR-022–TR-024 | ADR-0001, ADR-0003, ADR-0005 | redacted stable errors, digest/opaque provenance, source-free reports where applicable | **Implemented principle/evolving.** Actual PII authorization/retention belongs to owning product/service. |
| NVIDIA NIM/OpenCode, not Copilot token, for model automation | PR-008, PR-010 | TR-018, TR-024 | ADR-0005, ADR-0008 | repo/central automation contracts; PR #558 proposes hourly bounded caller and depends on central accepted workflow | **Active dependency.** Do not activate stale immutable pin before central protected-main acceptance. |
| Hourly autonomous PR/review loop | operational requirement | TR-024, TR-027–TR-030 | ADR-0008 | repository governance and active PR #558 for bounded review-repair caller | **Operational/evolving.** Waiting on review/checks must not stop unrelated executable work. |
| Security/SAST/fuzz/supply-chain gates | PR-010, PR-012 | TR-022–TR-024 | ADR-0008 | required CI/Security/SAST/fuzz/package gates, `SECURITY.md`/central workflows | **Implemented.** Do not weaken gates to shorten queue. |
| 100% meaningful coverage and docstrings | PR-012 | TR-025–TR-026 | ADR-0008 | CI coverage/docstring contracts across owned production surfaces | **Policy implemented/evolving with every feature.** Coverage must assert behavior, not line execution only. |
| Release / SBOM / provenance / rollback | PR-012 | TR-028–TR-030 | ADR-0008 | commercial release builder, acceptance, buyer/evidence/due-diligence artifacts | **Implemented baseline.** Avoid ambiguous valuation naming and bind claims to exact released artifact. |
| Database-object naming | PR-014 | TR-003, logical data model | ADR-0001 | no hosted DB owned here | **Not applicable to hosted persistence.** If reusable persistence is introduced: ADR first, two-or-more-word `snake_case`, migrations/rollback. |
| Internal name alignment | PR-014 | TR-029 | architecture authority map | current product is `fast-mlsirm`; ecosystem roles documented | **Continuous hygiene.** Deprecated/legacy internal names require compatibility-aware migration, not silent aliases. |
| PRD/TRD/ADR/Architecture/UML/ERD completeness | PR-014 | TR-030 + docs section | ADR index + root architecture + UML + logical ERD | this documentation-baseline branch | **Gap closed structurally by this PR; maintain with code.** |
| Buyer-facing hosted UX / Figma | downstream boundary | TR-001 | ADR-0001, UML ecosystem diagram | static buyer evidence exists in this repo; product workbench belongs to Psychometrics Commons | **Downstream.** Use Figma here only for reusable workflow/evidence design, not hosted UI ownership. |
| SOC 2 / CSAP / ISO/IEC 42001-minded controls | PR-010, PR-012 | TR-024, TR-030 | ADR-0005, ADR-0008 | security/provenance/release controls | **Design objective, not certification claim.** |

## 3. Research decisions captured from the project conversation

The following scientific conclusions are normative unless superseded by an ADR
and new evidence:

1. **Correlation is not parameter recovery or agreement.** Estimation accuracy is
   demonstrated with absolute error, bias, uncertainty/coverage and relevant
   response/information/decision recovery.
2. **Rater observations are not truth.** Humans and LLMs can be calibrated as
   fallible raters; severity, discrimination, range use and drift are distinct.
3. **Multidimensional, bifactor, higher-order, testlet, many-facet and latent-space
   structures answer different questions.** Do not use one as a fashionable
   substitute for design diagnosis.
4. **Latent space comes after substantive dimensions/testlets/facets when used as
   residual interaction structure.** Otherwise it can absorb omitted constructs.
5. **Formal model relation precedes formal model selection.** Boundary/nonregular
   models need appropriate reference distributions; non-nested models need
   distinguishability before a winner.
6. **Rotation has no universal criterion winner.** Use multi-start plus
   criterion-neutral recovery/stability/theory evidence.
7. **Dynamic rubric is a measurement-item lifecycle, not prompt-writing.**
   Candidate-blind benchmark criteria or cross-fitted discovery avoid circular
   evaluation; generated candidates must be screened/calibrated before use.
8. **Hierarchy and time are part of the model when the data-generating process has
   them.** Flattening them is a scientific error, not a performance optimization.
9. **Business/policy criticality is not psychometric discrimination.** A rare
   safety-critical failure may be a conjunctive policy gate even if it is not a
   high-information measurement item.

## 4. Documentation maintenance rule

Every material PR must answer:

- Does it change a PRD requirement or non-goal?
- Does it change TRD execution, API/schema/resource/error/security/release
  semantics?
- Does it establish or reverse an ADR decision?
- Does it alter the logical ERD or UML sequence/state/component diagram?
- Does it require new scientific doctoring/reference evidence?
- Does it change this traceability row's implementation status?

If yes, update the authoritative artifact in the same PR. Documentation debt is a
release blocker, but a documentation update is not a reason for the autonomous
maintenance loop to stop while executable code/PR/product work remains.
