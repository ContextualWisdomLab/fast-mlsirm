# Requirements Traceability Matrix

Baseline source: `docs/PRD.md`, `docs/TRD.md`, `ARCHITECTURE.md`, ADR-0001 through ADR-0009, and protected-main capability state recorded in `docs/architecture/capability_maturity.md`.

This matrix answers a different question from the capability maturity map. The PRD says **what the product must become**; this matrix shows **where each requirement is designed, who owns it, what evidence is required, and whether the current protected-main baseline already satisfies it**. A row marked partial or design-required is not a release claim.

## Functional requirements

| PRD requirement | Owner | Architecture / ADR authority | Current maturity | Minimum verification before full support |
|---|---|---|---|---|
| FR-01 Core psychometric computation | fast-mlsirm | `ARCHITECTURE.md`, TRD §3, ADR-0002 | IMPLEMENTED/PARTIAL by method | Rust/PyO3 tests, realistic method fixtures, recovery/fit evidence, packaging |
| FR-02 Rust-first numerical ownership | fast-mlsirm | TRD §1, ADR-0002 | IMPLEMENTED | delegation tests, no duplicate production math, wheel/reinstall, CPU/GPU parity where applicable |
| FR-03 Multilevel and time-aware measurement | fast-mlsirm contracts + TEPP for broader temporal analytics | TRD §4, ADR-0006 | PARTIAL / DESIGN-REQUIREMENT | connected-design checks, true-parameter bias/RMSE/coverage, irregular-time and multiple-membership simulations, explicit TEPP boundary |
| FR-04 Measurement-model selection | fast-mlsirm | TRD §3.3, ADR-0004 | PARTIAL | relation classifier, formal distinguishability, boundary-aware tests, cluster-aware holdout, true-structure selection simulation |
| FR-05 Bifactor scoreability | fast-mlsirm | TRD §3.4, ADR-0004 | PARTIAL | applicable ECV/PUC/omega-H/omega-HS/determinacy evidence, categorical-vs-latent-response labeling, score recovery |
| FR-06 Adaptive rotation | fast-mlsirm | TRD §3.5, ADR-0007 | PARTIAL | criterion gradient oracles, multi-start/stationarity, sign/permutation alignment, recovery/stability evidence |
| FR-07 Rubric and item-generation lifecycle | fast-mlsirm reusable contracts; host owns durable operation | TRD §5, ADR-0003, ADR-0009 | PARTIAL | closed-schema/replay tests, semantic screening, pilot/calibration evidence, linking, lifecycle transition tests |
| FR-08 AssessmentSpec and scoring contracts | fast-mlsirm | TRD §2/§6, ADR-0003 | IMPLEMENTED/PARTIAL | schema/fingerprint/version compatibility, provider-neutral adapters, replay/forgery tests |
| FR-09 Automated essay scoring validation | fast-mlsirm | `docs/UML.md` §4, TRD §6, ADR-0005 | IMPLEMENTED/PARTIAL | agreement, severity/fit/range, fairness/DIF, drift, adjudication and deterministic report evidence appropriate to the released surface |
| FR-10 Reference-free RAG measurement | fast-mlsirm reusable measurement contracts; contextual-orchestrator for model routing | `docs/UML.md` §5, TRD §7, ADR-0005/0009 | DESIGN-REQUIREMENT / partial research artifacts | accepted canonical schema, rater/testlet identification, perturbation/anchor design, model-family sensitivity, recovery/validity evidence |
| FR-11 Enterprise issue measurement | fast-mlsirm measurement adapters; consequential action layer downstream | `docs/UML.md` §7, TRD §8, ADR-0001 Decision 11 | IMPLEMENTED/PARTIAL | evidence/counterevidence provenance, uncertainty, stakeholder/rater structure; separate utility/causal validation for action ranking |
| FR-12 Reports and audit evidence | fast-mlsirm | Architecture quality/release, TRD §9/§12, ADR-0003/0008 | IMPLEMENTED/PARTIAL | deterministic replay, accessibility tests, redaction/non-reflection tests, stable IDs/fingerprints |
| FR-13 Release evidence | fast-mlsirm + `.github` control plane | TRD §11/§12, ADR-0008 | IMPLEMENTED/PARTIAL | exact-head CI/security/package/SBOM/provenance/reproducibility/recovery and accepted artifact hashes |

## Cross-cutting non-functional requirements

| Requirement | Owner | Authority | Verification |
|---|---|---|---|
| Scientific correctness | fast-mlsirm | ADR-0002/0004/0008, method doctoring | true-parameter bias/MAE/RMSE/coverage, convergence/failure classification, relation-safe comparison |
| Statement/branch/docstring completeness | fast-mlsirm | PRD §5, TRD §11, ADR-0008 | exact owned-production coverage and public documentation gates without meaningless exclusions |
| Performance/resource safety | fast-mlsirm | ADR-0002, method-specific resource doctoring | bounded allocations/subprocesses, environment-qualified benchmarks, concurrency tests |
| CPU/GPU parity | fast-mlsirm | ADR-0002 | explicit no-skip GPU evidence and numerical parity at declared tolerance |
| Security/supply chain | fast-mlsirm + `.github` | TRD §9/§11/§12, ADR-0008 | SAST, dependency/security scans, least privilege, immutable source/pins, SBOM/provenance |
| Privacy without destructive masking | host for direct identity/persistence; core for minimum-data contracts | Architecture data/privacy, TRD §9.2, ADR-0001 Decision 12 | purpose/authorization tests, opaque IDs, selective disclosure, no raw credential/source reflection, host retention/residency evidence |
| Accessibility | fast-mlsirm report surfaces; hosted UI downstream | TRD §13, ISO/IEC 40500:2025/WCAG 2.2 | report-specific WCAG/WAI-ARIA tests; hosted UI conformance remains downstream |
| Standalone + MSA interoperability | fast-mlsirm and consuming services | Architecture C4 boundary, ADR-0001 | package works without hosted product; explicit versioned interfaces; no cross-service DB reads/import cycles |
| Documentation consistency | fast-mlsirm | ADR-0001 Decision 14, `docs/documentation_coverage_matrix.md` | documentation contract tests, capability maturity classification, current links and changelog parity |

## Cross-repository ownership matrix

| Concern | Owner | fast-mlsirm relationship |
|---|---|---|
| Hosted assessment/session/consent/result lifecycle | Psychometrics Commons | downstream consumer of versioned measurement contracts |
| Physical product DB/migrations/tenant authorization | Psychometrics Commons | not owned here; `docs/ERD.md` is logical only |
| Identity/federation/passkeys | Keyverse | host-side identity boundary; core receives opaque/versioned subject references where needed |
| Broad temporal/event/relationship analytics | TEPP | consumes/produces versioned measurement-event artifacts; no duplicated general temporal platform |
| EMA/ESM client collection | Gyeot | downstream collection integration |
| Real-time LLM routing/orchestration | contextual-orchestrator | provider-neutral execution dependency when used; not core measurement semantics |
| Bulk asynchronous LLM execution | pg-llm-batch | downstream/adjacent batch execution service |
| Outbound SSRF/DNS/authority controls | EgressWeave | host/integration egress boundary |
| Organization CI/review/release control plane | ContextualWisdomLab/.github | reusable workflow/governance dependency; exact repository policy remains authoritative |
| Research catalog/release discovery | semantic-data-portal | downstream immutable research-release metadata/discovery |

## Traceability maintenance rule

A material PR must update this matrix when it changes a PRD requirement's owner, maturity, public contract, acceptance evidence, or downstream boundary. A feature is not promoted from PARTIAL/DESIGN-REQUIREMENT to IMPLEMENTED by documentation alone; the protected-main implementation and accepted exact-head evidence must change first.
