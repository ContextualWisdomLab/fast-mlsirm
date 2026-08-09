# Architecture documentation completeness and maintenance matrix

Status: **Authoritative maintenance audit**  
Last reviewed: 2026-08-09

This matrix answers whether the repository has enough durable documentation to reconstruct current product intent, technical boundaries, architecture, decisions, logical data relationships, threat model, scientific evidence and release obligations without mining chat history or PR bodies.

## Status vocabulary

- **IMPLEMENTED** — canonical document exists on this branch and describes protected-main behavior/policy without relying on unmerged code.
- **ACTIVE PR** — durable requirement/decision is known, but the corresponding runtime/scientific feature is still under open PR; documentation must not call it released.
- **PLANNED** — accepted/proposed direction with incomplete implementation/evidence.
- **DOWNSTREAM** — owned by Psychometrics Commons or another service; fast-mlsirm documents only the versioned boundary/handoff.
- **REJECTED/SUPERSEDED** — considered or historical design that must not silently return as current authority.

## Canonical documentation coverage

| Documentation capability | Before canonical baseline | Current target | Status | Maintenance / remaining gap |
|---|---|---|---|---|
| Product requirements | narrow/stale early-MVP summary plus feature plans | `docs/PRD.md` | IMPLEMENTED | update when buyer workflow/non-goal changes |
| Technical requirements | scattered agent/doctoring/spec rules | `docs/TRD.md` | IMPLEMENTED | update on numerical/runtime/security/release policy changes |
| Root architecture | no root architecture authority | `ARCHITECTURE.md` | IMPLEMENTED | keep current vs proposed explicit |
| Documentation authority/index | feature folders only | `docs/README.md` | IMPLEMENTED | new canonical categories must be linked here |
| Architecture decision log | decisions scattered across AGENTS/plans/PRs | `docs/adr/README.md`, ADRs | IMPLEMENTED | material decision needs status-bearing ADR/supersession |
| Component/UML views | no coherent canonical set | `docs/uml/*.puml` | IMPLEMENTED | update when module/dependency/lifecycle changes |
| Logical ERD | absent | `docs/erd/domain-model.puml` | IMPLEMENTED | remains logical/persistence-neutral; physical hosted DB is downstream |
| Requirements traceability | absent | `docs/traceability/requirements-matrix.md` | IMPLEMENTED | update maturity/evidence with material feature PRs |
| Scientific/standards basis | strong but scattered doctoring | `docs/traceability/research-basis.md` + doctoring | IMPLEMENTED | keep APA 7, primary/current standards and preprint status honest |
| Reusable-core threat model | implicit in security feature docs | `docs/security/threat-model.md` | IMPLEMENTED | update on new trust/native/provider/artifact boundaries |
| Documentation contract CI | absent | `tests/test_architecture_documentation_contract.py` | IMPLEMENTED | must fail on missing/stale/overclaiming architecture artifacts |
| Release/changelog evidence | managed changelog and release docs already exist | changelog fragment + existing release controls | IMPLEMENTED | render fragment before Ready/merge according to repo policy |
| Operational runbook for hosted product | intentionally not owned here | Psychometrics Commons/operator docs | DOWNSTREAM | link only when a versioned integration requires it |
| Physical DB schema/migrations | intentionally not owned here | Psychometrics Commons/owning host | DOWNSTREAM | do not manufacture ORM from logical ERD |
| Tenant/RBAC/SSO/SCIM/UI/billing | not a reusable core concern | hosted product/services | DOWNSTREAM | retain dependency direction only |

## Conversation-wide scientific/product coverage

| Work family | Documentation maturity | Runtime/evidence maturity | State |
|---|---|---|---|
| Fallible human/LLM raters and many-facet calibration | PRD/TRD + ADR-0005 + traceability | baseline facets/scoring exists; generalized discrimination/range/drift remains incremental | IMPLEMENTED / PLANNED extensions |
| Correlation is not parameter recovery/agreement | ADR-0008 + PRD/TRD + research basis | recovery/simulation evidence exists across model families | IMPLEMENTED governance |
| Reference-free RAG measurement | PRD/TRD + traceability | no single canonical end-to-end RAG observation adapter/bank workflow on protected main | PLANNED |
| Dynamic evidence-grounded rubric generation | PRD/TRD + ADR-0003/0004 + item UML/ERD | strong rubric/generation/audit/pilot primitives exist | IMPLEMENTED primitives / PLANNED closed loop |
| Governed item-bank lifecycle | ADR-0004 + state diagram | pilot/admission/lifecycle pieces exist; unified approve/active/link/drift/exposure/retire workflow remains incomplete | PLANNED/partial |
| Bifactor / higher-order / testlet / two-tier / many-facet relation | ADR-0006 + model-selection UML/research basis | family-specific features/evidence vary | IMPLEMENTED policy / partial family coverage |
| Latent-space residual interaction | architecture/PRD/TRD | supported model family exists, but must follow substantive dimension/testlet/facet diagnosis | IMPLEMENTED with interpretation gate |
| Formal non-nested distinguishability | ADR-0006 + traceability | fail-closed comparison exists where formal family inputs are incomplete; full score/information metadata still needed | PLANNED extension |
| Adaptive rotation criterion selection | ADR-0009 + PRD/TRD | active feature work must earn protected-main integration | ACTIVE PR / PLANNED release |
| Multilevel/multiple-membership/cross-classified contracts | ADR-0007 + UML/ERD/PRD/TRD | active PR exists; dedicated namespace not accepted until protected-main evidence | ACTIVE PR |
| Temporal/longitudinal/drift models | ADR-0007 + PRD/TRD | design/primitives exist; continuous-time estimator claims require separate recovery | PLANNED/partial |
| Automated essay scoring calibration/validation | PRD/TRD + ADR-0005 | governed essay contracts/calibration/validation/reporting exist; rater-range/discrimination/drift extensions remain active | IMPLEMENTED baseline / ACTIVE extensions |
| Enterprise issue measurement | PRD/TRD + traceability | reusable evidence/calibration adapters exist | IMPLEMENTED measurement; causal intervention utility DOWNSTREAM/policy |
| Factor retention | PRD/TRD + ADR-0006 | diagnostics exist; unified evidence API remains a gap | PLANNED extension |
| Rust-first numerical core / CPU+GPU parity | ADR-0002 + TRD | current model-specific support/evidence varies by kernel | IMPLEMENTED architecture, feature-specific evidence required |
| Canonical PyO3/public-export registry | ADR-0011 | current exports work, but future feature PRs must converge rather than creating competing initialization schemes | PLANNED hardening |
| PII/purpose limitation | ADR-0012 + threat model + PRD/TRD | source-free/digest-based contracts exist where possible; hosted access/retention is downstream | IMPLEMENTED reusable policy / DOWNSTREAM operations |
| LLM orchestration/model credentials | ADR-0010 | repository/org automation policy exists | IMPLEMENTED governance |

## P0 documentation gaps

A P0 gap blocks treating the architecture package as complete:

- missing canonical PRD or TRD;
- missing root architecture boundary;
- missing ADR index/status for a material cross-cutting decision;
- missing UML/ERD view for a major public-contract or lifecycle change;
- missing reusable-core threat model after a new trust boundary;
- traceability that falsely marks an active/planned feature as protected-main implemented;
- a stale historical summary that competes with the canonical requirements source; or
- architecture claims that move hosted product DB/HTTP/tenant/RBAC ownership into fast-mlsirm without a superseding ADR.

## P1 documentation gaps

P1 gaps do not automatically block unrelated development, but must be repaired before release of the affected capability:

- missing method-specific doctoring/primary source;
- missing recovery/scoreability interpretation boundary;
- missing failure/recovery/rollback instructions for a changed public artifact;
- missing privacy/security abuse case for new provider/native/artifact surfaces; or
- missing changelog/release evidence for a user-visible accepted capability.

## P2 improvements

- richer rendered architecture diagrams/site navigation;
- downstream hosted-workbench Figma/UX links;
- automated link/PlantUML rendering checks beyond the current source contract;
- generated traceability views from contract metadata; and
- buyer/operator views that consume these artifacts without becoming a second source of truth.

## Maintenance gate

Every material PR should answer:

1. Did product requirements or non-goals change?
2. Did a technical invariant/trust/resource/release rule change?
3. Did a durable architecture/scientific decision change or need supersession?
4. Did component/data/lifecycle/deployment/ERD views change?
5. Did the threat model gain a new asset/actor/abuse case?
6. Did an implementation maturity state change (PLANNED/ACTIVE -> protected-main IMPLEMENTED)?
7. Did source/test evidence change enough that traceability is stale?
8. Is the changelog/release evidence synchronized?

If yes, update the corresponding canonical document in the same PR or record a precise downstream/no-change justification. Documentation drift is treated as a repository defect rather than post-release cleanup.
