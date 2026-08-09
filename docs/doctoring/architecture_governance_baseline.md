# Architecture Governance Doctoring

Date reviewed: 2026-08-09  
Scope: canonical PRD/TRD/architecture/ADR/UML/ERD/traceability baseline for `fast-mlsirm`.

## Decision record

The repository previously had strong method, feature, agent, scoring-contract, rubric, and release-evidence documentation but did not have one canonical architecture/requirements spine. The baseline introduces requirements, architecture viewpoints, durable ADRs, a logical ERD, UML/behavioral views, and requirement-to-evidence traceability while keeping implementation maturity explicit.

The documents are intentionally organized as complementary information items rather than one monolithic design document:

- PRD — product requirements and non-goals;
- TRD — enforceable technical/scientific requirements;
- Architecture — system-of-interest, stakeholders, viewpoints, boundaries and fitness functions;
- ADR — durable decisions and rejected alternatives;
- UML — structural/behavioral/state views;
- logical ERD — information relationships without assigning a physical database to the core;
- traceability — requirement-to-implementation/evidence maturity;
- feature/method doctoring — detailed scientific and algorithmic evidence.

## Standards evidence

The following standards were checked against official publisher/standards-body information before establishing the baseline. Standard text is not copied into this repository; the documentation applies their concepts at a high level.

### Architecture descriptions

ISO/IEC/IEEE 42010:2022 defines requirements for architecture descriptions of software, systems, and enterprises. The baseline therefore identifies the system of interest, stakeholders/concerns, viewpoints/views, decisions, and correspondence across product/technical/information/decision artifacts.

### Requirements engineering

ISO/IEC/IEEE 29148:2018 remains the governing published requirements-engineering standard while a revision project is in progress. The PRD/TRD use uniquely identifiable requirements and acceptance/evidence boundaries to make downstream traceability possible.

### Software life-cycle processes

ISO/IEC/IEEE 12207:2026 is the current software-life-cycle process standard. The architecture baseline treats implementation, verification, review, release, maintenance, and documentation as lifecycle evidence rather than equating a merged feature with release readiness.

### Life-cycle information items

ISO/IEC/IEEE 15289:2019 remains the published life-cycle information-item standard confirmed current in its most recent review. The baseline separates requirements, architecture, decisions, method evidence, verification evidence, and release evidence rather than treating a README as the single authoritative information item.

### Product quality

ISO/IEC 25010:2023 defines the product-quality model used to frame technical quality requirements such as functional suitability, performance efficiency, compatibility, reliability, security, maintainability, portability, and relevant interaction/accessibility properties.

### AI lifecycle and management-system concerns

ISO/IEC 5338:2023 provides an AI-system lifecycle process framework relevant to optional LLM-backed evaluation/orchestration. ISO/IEC 42001:2023 informs management-system concerns such as AI lifecycle governance and evidence, but this repository does **not** claim ISO/IEC 42001 certification.

### UML

OMG UML 2.5.1 is the stable UML specification used as the conceptual vocabulary for component, class, sequence, activity/state, and deployment views. Mermaid is the rendering notation used in-repository; it is not claimed to be an OMG UML interchange implementation.

## Security/compliance positioning

The architecture is designed to produce evidence useful for enterprise assurance, SOC 2 control environments, and Korean public/cloud security-assurance planning such as CSAP-relevant deployments, but `fast-mlsirm` itself does not claim certification. Hosted identity, tenant isolation, cloud deployment, physical database, key management, data residency, incident operations, and customer data-rights controls belong to the owning product/runtime and therefore must be evidenced there.

The privacy architecture deliberately does not impose blanket PII masking that would destroy required measurement/operational semantics. Instead it uses purpose-bound data minimization in governed artifacts, content fingerprints/opaque references for provenance, and leaves raw operational data under the owning service's authorization, encryption, retention, export/deletion, and audit controls.

## Scientific architecture evidence carried forward

The baseline preserves the research conclusions already established in method-specific doctoring:

- LLM/human judges are fallible raters rather than ground truth;
- reference-free evaluation distinguishes groundedness from world correctness and completeness;
- automated scoring validity cannot be established by correlation alone;
- parameter recovery uses scale/alignment followed by bias/RMSE/coverage/convergence evidence;
- correlated MIRT, bifactor, higher-order, testlet, two-tier, many-facet, and latent-space models answer distinct structural questions;
- formal model comparison must match the actual nested/boundary/non-nested relationship;
- bifactor fit and scoreability are separate decisions;
- no universal factor-rotation criterion or finite-multistart global-optimum claim is permitted;
- multilevel, cross-classified, multiple-membership, rater, testlet, and temporal structure must be represented when present in the data-generating design;
- production mathematical/psychometric arithmetic remains Rust-first, with CPU/GPU evidence appropriate to the claimed backend.

## Documentation falsification criteria

This architecture baseline must be revised or superseded if protected-main evidence shows any of the following:

1. `fast-mlsirm` acquires mandatory hosted-product/database/auth dependencies;
2. Python becomes the production owner of a numerical feature documented as Rust-owned;
3. public contract identity no longer follows the stated version/fingerprint semantics;
4. an implemented model requires a different relation/identification assumption than the current ADR;
5. item-bank lifecycle ownership moves into a different repository/bounded context;
6. temporal contracts begin using elapsed intervals in the likelihood while documentation still calls the model discrete occasion-step;
7. a new regulatory/compliance claim is made without owning-system evidence;
8. PRD/TRD traceability marks a planned feature as implemented without protected-main code/tests.

## References — APA 7th

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

International Organization for Standardization. (2023). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software Quality Requirements and Evaluation (SQuaRE)—Product quality model*.

International Organization for Standardization. (2023). *ISO/IEC 42001:2023 Information technology—Artificial intelligence—Management system*.

International Organization for Standardization. (2023). *ISO/IEC 5338:2023 Information technology—Artificial intelligence—AI system life cycle processes*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2018). *ISO/IEC/IEEE 29148:2018 Systems and software engineering—Life cycle processes—Requirements engineering*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2019). *ISO/IEC/IEEE 15289:2019 Systems and software engineering—Content of life-cycle information items (documentation)*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2026). *ISO/IEC/IEEE 12207:2026 Systems and software engineering—Software life cycle processes*.

Object Management Group. (2017). *OMG Unified Modeling Language (OMG UML), Version 2.5.1*.
