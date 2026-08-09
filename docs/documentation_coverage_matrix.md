# Documentation Coverage Matrix

This matrix evaluates whether repository documentation is sufficient to reconstruct the current product, scientific boundaries, software architecture, data contracts, and commercialization/release obligations without reading implementation code first.

## Executive assessment

Before the documentation-baseline PR, documentation was **not sufficient as a coherent system specification** even though many high-quality method-specific RFCs, doctoring records, plans, release guides, buyer-evidence documents and tests existed.

The largest defects were:

1. `docs/prd_trd_summary.md` described an early NumPy-first MLS2PLM MVP and contradicted the current Rust-first production direction.
2. There was no authoritative root `ARCHITECTURE.md` describing the current repository boundary with Psychometrics Commons and other CWL services.
3. PRD and TRD were combined in a short stale summary instead of separating product outcomes from implementation requirements.
4. UML/component/sequence views were scattered or absent; there was no coherent end-to-end view of rubric → item → calibration, automated scoring, RAG evaluation, enterprise measurement, or multilevel/longitudinal flows.
5. No persistence-neutral ERD documented how reusable contracts map into downstream durable stores while keeping ORM/database ownership outside the core.
6. Architectural decisions were distributed across AGENTS/CLAUDE, method docs, PR descriptions and Superpowers plans rather than being captured in an authoritative ADR baseline.
7. Compliance/privacy requirements did not have one architecture decision explaining why blanket PII masking is inappropriate for some psychometric workflows and which alternative controls are expected.
8. The relationship among model retention, correlated MIRT, bifactor, higher-order, testlet, two-tier, multifaceted and latent-space structures was documented in research notes but not in the system architecture baseline.

This PR addresses those baseline gaps. Method-specific documentation remains authoritative for detailed formulas and implementation evidence.

## Coverage table

| Documentation concern | Pre-baseline state | Baseline document | Current judgment | Follow-up trigger |
|---|---|---|---|---|
| Product vision / users / buyer outcomes | Fragmented across README and commercial docs | `docs/PRD.md` | Sufficient baseline | Update when a new bounded product capability becomes supported/unsupported |
| Product non-goals / repository scope | Partially in AGENTS/CLAUDE | `docs/PRD.md`, `ARCHITECTURE.md`, ADR-0001 | Sufficient baseline | Update when responsibility moves between fast-mlsirm and Psychometrics Commons/services |
| Technical runtime architecture | Stale NumPy-first summary | `docs/TRD.md`, `ARCHITECTURE.md` | Sufficient baseline | Update on numerical-backend/PyO3/package architecture changes |
| Rust/Python numerical ownership | Present but distributed | `docs/TRD.md`, ADR-0001 | Sufficient baseline | Any new production numerical module |
| CPU/GPU parity contract | Distributed across feature docs | `docs/TRD.md`, ADR-0001 | Sufficient baseline | New GPU backend/kernel |
| Factor retention vs structure selection | Research docs only | `ARCHITECTURE.md`, `docs/TRD.md`, ADR-0001 | Sufficient architecture baseline | Formal comparison API or relationship classifier changes |
| Bifactor scoreability | Method-specific docs + tests | Baseline plus method docs | Sufficient at architecture level | New categorical/observed-score reliability estimator or thresholds |
| Rotation architecture | Method-specific plan/doc | Baseline plus rotation docs | Sufficient at architecture level | New criterion family, optimizer or GPU multi-start backend |
| Multilevel / multiple-membership | Emerging implementation docs | `ARCHITECTURE.md`, `docs/TRD.md`, UML | Sufficient contract baseline | First accepted Rust contextual/longitudinal estimator |
| Temporal / longitudinal semantics | Emerging implementation docs | `ARCHITECTURE.md`, `docs/TRD.md`, ADR-0001 | Sufficient contract baseline | Continuous-time model or drift state-model release |
| Rubric → item-generation lifecycle | Several RFCs/plans | `docs/PRD.md`, `docs/TRD.md`, UML | Sufficient system baseline | Governed item bank becomes production-stable |
| AssessmentSpec / scoring contracts | Existing method docs | PRD/TRD/UML | Sufficient system baseline | Contract schema version change |
| Automated essay evaluation | Existing scoring/essay docs | PRD/TRD/UML | Sufficient architecture baseline | New scorer/feedback model or consequential use contract |
| Reference-free RAG measurement | Research notes, not system baseline | PRD/TRD/UML/ADR | Sufficient design baseline | Canonical RAG observation schema stabilizes |
| Enterprise issue measurement | Existing vertical docs | PRD/TRD/UML/ADR | Sufficient design baseline | Decision-utility module becomes supported product API |
| Persistence / logical entities | No authoritative ERD | `docs/ERD.md` | Sufficient persistence-neutral baseline | Downstream host adopts versioned persistence contract |
| Hosted product / MSA boundaries | Recently added to AGENTS/CLAUDE | `ARCHITECTURE.md`, ADR-0001 | Sufficient baseline | Any cross-repository ownership change |
| Identity / PII strategy | Scattered security requirements | TRD, Architecture, ADR-0001, ERD | Sufficient architecture baseline | New regulated/high-sensitivity deployment profile |
| CSAP / SOC 2 readiness intent | General project instruction | TRD/ADR baseline | Directionally sufficient; not certification evidence | Add formal control mapping when hosted product/control plane is assessed |
| AI management / risk governance | Scattered | TRD/ADR baseline | Sufficient architecture baseline | NIST AI RMF revision or ISO control-profile change |
| Release / SBOM / provenance | Strong existing release docs | PRD/TRD/Architecture | Strong | Version/release pipeline contract changes |
| Accessibility of reports | Method-specific doctoring | PRD/TRD/Architecture | Strong for implemented report surfaces | New UI/workbench or dynamic hosted interface |
| UML context/component/sequence views | No coherent set | `docs/UML.md` | Sufficient baseline | New major bounded context or end-to-end workflow |
| ERD | Absent | `docs/ERD.md` | Sufficient logical baseline | Persistence contract becomes normative |
| ADR index / governing decisions | Fragmented | ADR-0001 | Baseline created; additional ADRs needed for material future decisions | Each irreversible/cross-cutting architectural decision |
| PRD/TRD index | Stale combined summary | `docs/prd_trd_summary.md` index | Corrected | Keep as index only |

## Residual documentation gaps to close through normal development

The baseline does **not** mean all documentation work is complete. The following should be added when the corresponding implementation becomes stable enough to avoid speculative contracts:

### 1. Canonical PyO3 registry ADR

Create a dedicated ADR when bifactor, rotation and future secondary modules share one final registry/initialization pattern across Linux/macOS/Windows wheels.

### 2. Formal model-comparison mathematics ADR/TRD supplement

When full Vuong distinguishability and boundary-aware comparison APIs stabilize, document score-vector/information-matrix contracts, clustering unit, bootstrap semantics and decision statuses with equation-to-source traceability.

### 3. Governed item-bank lifecycle state machine

When the bank is implemented, add a normative state-machine diagram covering draft → audited → screened → pilot → calibrated → approved → active → suspended/quarantined → retired, including approval, rollback, linking and exposure semantics.

### 4. Reference-free RAG canonical schema

Add a schema-level document once the canonical query/evidence/claim/criterion/judge/system-run/testlet/anchor objects are accepted into the public package.

### 5. Multilevel/longitudinal estimator architecture

The current baseline describes representation and scientific constraints. The first production Rust estimator for cross-classified/multiple-membership/longitudinal structure must add estimator-specific objective, integration, identification, GPU/CPU and recovery diagrams.

### 6. Compliance control mappings

`fast-mlsirm` can document reusable controls, but formal CSAP/SOC 2 control matrices for hosted operations belong primarily to Psychometrics Commons and the organization control plane. Add cross-repository mapping only when control ownership and evidence locations are stable.

### 7. Buyer-facing workbench information architecture

Figma/Product Design should be updated when backend contracts for the governed Rubric → Blueprint → Candidate → Screening → Pilot → Calibration → Item Bank workflow are stable. Avoid designing a UI around temporary schemas.

## Documentation acceptance rule

A documentation set is considered sufficient for a release-changing feature only when an independent reviewer can answer, without reading implementation code first:

1. What user problem and bounded product capability changed?
2. Which repository owns the capability and what is explicitly outside its scope?
3. What data/contracts cross the boundary?
4. Which mathematical layer owns each calculation?
5. What model assumptions and interpretation limits apply?
6. How are multilevel/time/context effects preserved when relevant?
7. What security/privacy/PII controls apply without destroying valid measurement?
8. What realistic tests/recovery/fairness/compatibility evidence is required?
9. What migration/rollback/version/linking implications exist?
10. What exact-head CI/security/package/review/release evidence is required?

If any answer is unclear or contradictory across PRD/TRD/Architecture/ADR/UML/ERD/method doctoring, documentation is a release blocker rather than optional cleanup.
