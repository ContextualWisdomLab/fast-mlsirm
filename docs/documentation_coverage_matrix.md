# Documentation Coverage Matrix

This matrix evaluates whether repository documentation is sufficient to reconstruct the current product, scientific boundaries, software architecture, data contracts, implementation maturity, cross-repository ownership and commercialization/release obligations without reading implementation code first.

## Executive assessment

Before this documentation-baseline PR, documentation was **not sufficient as a coherent system specification** even though many high-quality method-specific RFCs, doctoring records, plans, release guides, buyer-evidence documents and tests existed.

The largest defects were:

1. `docs/prd_trd_summary.md` described an early NumPy-first MLS2PLM MVP and contradicted the current Rust-first production direction.
2. There was no authoritative root `ARCHITECTURE.md` describing the current repository boundary with Psychometrics Commons and other CWL services.
3. PRD and TRD were combined in a short stale summary instead of separating product outcomes from implementation requirements.
4. UML/component/sequence views were scattered or absent; there was no coherent end-to-end view of rubric → item → calibration, automated scoring, RAG evaluation, enterprise measurement, or multilevel/longitudinal flows.
5. No persistence-neutral ERD documented how reusable contracts map into downstream durable stores while keeping ORM/database ownership outside the core.
6. Architectural decisions were distributed across AGENTS/CLAUDE, method docs, PR descriptions and Superpowers plans rather than being captured in a durable ADR set with index/template/supersession rules.
7. Compliance/privacy requirements did not have one architecture decision explaining why blanket PII masking is inappropriate for some psychometric workflows and which alternative controls are expected.
8. The relationship among model retention, correlated MIRT, bifactor, higher-order, testlet, two-tier, multifaceted and latent-space structures was documented in research notes but not in the system architecture baseline.
9. Target architecture and shipped capability were not separated, creating a risk that planned multilevel/RAG/item-bank behavior could be read as already implemented.
10. There was no requirement→owner→ADR/architecture→maturity→verification matrix tying PRD requirements to implementation evidence.

This PR addresses those baseline gaps. Method-specific documentation remains authoritative for detailed formulas and feature evidence, while `docs/architecture/capability_maturity.md` prevents architecture requirements from being promoted to shipped claims.

## Coverage table

| Documentation concern | Pre-baseline state | Baseline document | Current judgment | Follow-up trigger |
|---|---|---|---|---|
| Product vision / users / buyer outcomes | Fragmented across README and commercial docs | `docs/PRD.md` | Sufficient baseline | Update when a bounded product capability requirement changes |
| Product non-goals / repository scope | Partially in AGENTS/CLAUDE | `docs/PRD.md`, `ARCHITECTURE.md`, ADR-0001 | Sufficient baseline | Any responsibility move between fast-mlsirm and Psychometrics Commons/services |
| Requirement traceability | Absent | `docs/requirements_traceability.md` | Sufficient baseline | Any FR/NFR owner, maturity, contract or acceptance-evidence change |
| Capability maturity / shipped-vs-target distinction | Absent | `docs/architecture/capability_maturity.md` | Sufficient baseline | Every material protected-main capability promotion/demotion |
| Technical runtime architecture | Stale NumPy-first summary | `docs/TRD.md`, `ARCHITECTURE.md` | Sufficient baseline | Numerical-backend/PyO3/package architecture change |
| Rust/Python numerical ownership | Present but distributed | TRD, ADR-0002 | Sufficient baseline | Any new production numerical module or fallback policy |
| CPU/GPU parity contract | Distributed across feature docs | TRD, ADR-0002 | Sufficient baseline | New GPU backend/kernel or fallback semantics |
| Factor retention vs structure selection | Research docs only | Architecture, TRD, ADR-0004 | Sufficient architecture baseline; implementation remains partial | Formal comparison API or relationship classifier changes |
| Bifactor scoreability | Method-specific docs + tests | TRD, ADR-0004, capability map + method docs | Sufficient at architecture level | New categorical/observed-score reliability estimator or interpretation contract |
| Rotation architecture | Method-specific plan/doc | TRD, ADR-0007 + rotation docs | Sufficient at architecture level | New criterion family, optimizer or GPU multi-start backend |
| Multilevel / multiple-membership | Emerging implementation docs | Architecture, TRD, ADR-0006, UML, capability map | Sufficient contract baseline; general estimator not claimed | First accepted Rust contextual/multiple-membership estimator |
| Temporal / longitudinal semantics | Emerging implementation docs | Architecture, TRD, ADR-0006, capability map | Sufficient contract baseline; continuous-time remains design-required | Continuous-time or new drift state-model release |
| Rubric → item-generation lifecycle | Several RFCs/plans | PRD, TRD, UML, ADR-0003/0009, capability map | Sufficient system baseline; lifecycle implementation partial | Bank lifecycle becomes production-stable or owner changes |
| AssessmentSpec / scoring contracts | Existing method docs | PRD/TRD/UML/ADR-0003 | Sufficient system baseline | Contract schema/version migration |
| Automated essay evaluation | Existing scoring/essay docs | PRD/TRD/UML/ADR-0005 + capability map | Sufficient architecture baseline | New scorer/rater model, consequential use or fairness/range/drift contract |
| Reference-free RAG measurement | Research notes, not system baseline | PRD/TRD/UML/ADR-0005/0009 + capability map | Sufficient design baseline; public canonical schema not claimed | Canonical RAG observation schema stabilizes |
| Enterprise issue measurement | Existing vertical docs | PRD/TRD/UML/ADR-0001 + traceability | Sufficient design baseline | Decision-utility module becomes a supported public API |
| Persistence / logical entities | No authoritative ERD | `docs/ERD.md` | Sufficient persistence-neutral baseline; physical DB explicitly N/A | Downstream host adopts/version-changes a persistence contract |
| ERD relationship integrity | Absent | `docs/ERD.md` + documentation contract test | Sufficient baseline | Logical entity/FK/cardinality change |
| Hosted product / MSA boundaries | Recently added to AGENTS/CLAUDE | Architecture, ADR-0001, traceability | Sufficient baseline | Any cross-repository ownership change |
| Identity / PII strategy | Scattered security requirements | TRD, Architecture, ADR-0001, ERD | Sufficient architecture baseline | New regulated/high-sensitivity deployment profile or owner change |
| CSAP / SOC 2 readiness intent | General project instruction | TRD/ADR baseline | Directionally sufficient; not certification evidence | Add formal control mapping when hosted product/control plane evidence ownership stabilizes |
| AI management / risk / impact governance | Scattered | TRD/ADR baseline | Sufficient architecture baseline | NIST AI RMF revision or ISO governance/risk/impact standard change |
| Current accessibility standard | WCAG references scattered | TRD/Architecture | Sufficient baseline using ISO/IEC 40500:2025/WCAG 2.2 | W3C/ISO publication change or new hosted UI surface |
| Release / SBOM / provenance | Strong existing release docs | PRD/TRD/ADR-0008/Architecture | Strong | Version/release pipeline contract changes |
| Accessibility of reports | Method-specific doctoring | PRD/TRD/Architecture | Strong for implemented report surfaces | New UI/workbench/dynamic hosted interface |
| UML context/component/sequence/deployment views | No coherent set | `docs/UML.md` | Sufficient baseline | New major bounded context or end-to-end workflow |
| ERD | Absent | `docs/ERD.md` | Sufficient logical baseline; physical schema not owned here | Persistence contract becomes normative downstream |
| ADR lifecycle / template | Absent | `docs/adr/README.md`, `docs/adr/0000-template.md` | Sufficient baseline | ADR governance requirements change |
| Durable governing decisions | Fragmented | ADR-0001 through ADR-0009 | Sufficient baseline decomposition | Each irreversible/cross-cutting architectural decision or supersession |
| PRD/TRD index | Stale combined summary | `docs/prd_trd_summary.md` index | Corrected | Keep as index only |

## Residual documentation gaps to close through normal development

The baseline does **not** mean all documentation work is complete. The following should be added only when the corresponding implementation/decision is stable enough to avoid speculative contracts:

### 1. Final PyO3 registry ADR

The architecture requires a single maintainable composed registration/export strategy, but the final implementation decision should receive a dedicated ADR only after the competing integration approach is accepted on protected main. Do not create an ADR merely to claim the gap is closed.

### 2. Formal model-comparison mathematics ADR/TRD supplement

When full Vuong distinguishability and boundary-aware comparison APIs stabilize, document score-vector/information-matrix contracts, clustering unit, bootstrap semantics, relation statuses and decision thresholds with equation-to-source traceability.

### 3. Implemented item-bank lifecycle state machine

ADR-0009 records the accepted lifecycle. When the bank runtime is implemented, add a normative state-machine diagram tied to actual transition APIs covering draft → audited → screened → pilot → calibrated → approved → active → suspended/quarantined → retired, including authorization, rollback, linking and exposure semantics.

### 4. Reference-free RAG canonical schema

Add a schema-level document once canonical query/evidence/claim/criterion/judge/system-run/testlet/anchor objects are accepted into the public package. Until then, capability maturity remains DESIGN-REQUIREMENT rather than pretending architecture prose is an API.

### 5. Multilevel/longitudinal estimator architecture

The current baseline describes representation and scientific constraints. The first production Rust estimator for cross-classified/multiple-membership/longitudinal structure must add estimator-specific objective, integration/optimization, identification, CPU/GPU and recovery diagrams.

### 6. Compliance control mappings

`fast-mlsirm` can document reusable controls, but formal CSAP/SOC 2 control matrices for hosted operations belong primarily to Psychometrics Commons and the organization control plane. Add a cross-repository control/evidence matrix only when control owners and evidence locations are stable.

### 7. Buyer-facing workbench information architecture

Figma/Product Design should be updated when backend contracts for the governed Rubric → Blueprint → Candidate → Screening → Pilot → Calibration → Item Bank workflow are stable. Avoid designing a UI around temporary schemas.

### 8. Threat-model document if attack surface materially expands

Current security/privacy architecture is sufficient for the package baseline. Add a dedicated threat model when a new native parser, network/provider boundary, hosted execution surface, persistent trust domain or high-sensitivity data path materially changes attack paths.

## Documentation acceptance rule

A documentation set is considered sufficient for a release-changing feature only when an independent reviewer can answer, without reading implementation code first:

1. What user problem and bounded product capability changed?
2. Which repository owns the capability and what is explicitly outside its scope?
3. Is the capability implemented, partial, design-required, downstream-owned, or not applicable?
4. What data/contracts cross the boundary?
5. Which mathematical layer owns each calculation?
6. What model assumptions and interpretation limits apply?
7. How are multilevel/time/context effects preserved when relevant?
8. What security/privacy/PII controls apply without destroying valid measurement?
9. What realistic tests/recovery/fairness/compatibility evidence is required?
10. What migration/rollback/version/linking implications exist?
11. What exact-head CI/security/package/review/release evidence is required?
12. Which ADR governs the decision and what objective condition would supersede it?

If any answer is unclear or contradictory across PRD/TRD/Architecture/ADRs/UML/ERD/capability maturity/traceability/method doctoring, documentation is a release blocker rather than optional cleanup.
