# Architecture documentation completeness and maintenance matrix

Status: **Authoritative maintenance audit**  
Last reviewed: 2026-08-16

This matrix answers whether GitHub can reconstruct the current `fast-mlsirm` product, technical, scientific, security, operability, and release intent without relying on chat history or stale pull-request prose. File existence is not sufficient: document availability and product-capability maturity are evaluated separately against protected-main code, tests, workflows, accepted decisions, and live work.

## Status vocabulary

### Documentation-family state

- **PRESENT_CURRENT** — canonical artifact exists on protected main and is consistent with current protected-main ownership and policy.
- **PRESENT_STALE** — artifact exists but materially contradicts current protected-main behavior, ownership, version, or accepted policy.
- **PARTIAL** — artifact exists and is directionally correct but lacks material coverage needed for the affected release or buyer claim.
- **MISSING** — a required canonical artifact has no discoverable equivalent.
- **NOT_APPLICABLE** — the family is intentionally not owned here and the ownership boundary is documented.
- **SUPERSEDED** — historical artifact is retained only for compatibility/history and is not authoritative.
- **OWNED_BY_ACTIVE_PR** — a non-protected-main branch currently owns a coherent update; the branch must not be described as shipped truth.

### Capability maturity

- **IMPLEMENTED_ON_PROTECTED_MAIN** — accepted source/tests/contracts are ancestral to protected main.
- **IMPLEMENTED_ON_ACTIVE_PR** — implementation exists only on a current open PR.
- **PARTIAL** — useful protected-main primitives exist but the declared end-to-end capability is incomplete.
- **ACCEPTED_ARCHITECTURE** — durable design is accepted while implementation remains incomplete.
- **PLANNED** — desired work is known but not accepted as implemented.
- **RESEARCH_ONLY** — evidence or exploration exists without a production contract.
- **DOWNSTREAM** — owned by `ContextualWisdomLab/psychometrics-commons` or another explicit host/service.
- **SUPERSEDED** — a previous capability/implementation path has been replaced.
- **REJECTED** — reviewed and intentionally excluded.
- **OUT_OF_SCOPE** — outside the reusable measurement-core boundary.

For compatibility with review reports, the human-readable status shorthand is
also normative: **IMPLEMENTED**, **ACTIVE PR**, **PLANNED**, and
**DOWNSTREAM**. The phrase **IMPLEMENTED / PLANNED extensions** means that a
protected-main primitive exists while the explicitly named broader extension
remains planned or partial; **ACTIVE PR** never means protected-main truth.
The **Canonical PyO3/public-export registry** is the target source of truth
for native entrypoints. ADR-0011 remains **Proposed**: protected main still
uses its existing separate native initializers and package export paths, while
**ACCEPTED_ARCHITECTURE / PARTIAL** describes the target registry capability
that is not yet a protected-main completion claim.

## Canonical documentation coverage

| Documentation capability | Canonical target | State | Current fitness / maintenance rule |
|---|---|---|---|
| Product requirements | `docs/PRD.md` | PRESENT_CURRENT | update when users/JTBD, product scope, non-goals, buyer acceptance, or downstream ownership changes |
| Technical requirements | `docs/TRD.md` | PRESENT_CURRENT | update on numerical authority, public-contract, runtime, resource, security, privacy, interoperability, or release-rule changes |
| Root architecture | `ARCHITECTURE.md` | PRESENT_CURRENT | protected-main architecture baseline is integrated; keep protected-main vs active-PR/planned behavior explicit |
| Documentation authority/index | `docs/README.md` | PRESENT_CURRENT | all canonical families and strong equivalents must remain discoverable from one graph |
| Architecture decisions | `docs/adr/README.md`, status-bearing ADRs | PRESENT_CURRENT | durable cross-cutting decisions require explicit status and supersession rather than silent prose edits |
| Continuous execution governance | ADR-0013 + architecture/TRD links | PRESENT_CURRENT | protected-main governance describes RCA, feasibility, work conservation, writer leases, evidence freshness, and final-sweep behavior; scheduler runtime remains external to package capability |
| Standards status/watch | `docs/standards_watch.md` | PRESENT_CURRENT | published editions remain normative; drafts/revision projects remain watch items and never imply certification |
| Verification / test strategy | `docs/verification_validation_plan.md` | PRESENT_CURRENT | serves the Test Strategy/V&V role; update for new estimators, scorers, generalization claims, recovery evidence, security boundaries, or release evidence |
| UML/component and behavioral views | `docs/uml/*.puml` | PARTIAL | current component/deployment/scoring/model-selection/item-lifecycle views are useful; maintain a complete indexed inventory and add authority/recovery/public-contract class views when materially needed |
| Logical ERD / evidence model | `docs/erd/domain-model.puml` | PARTIAL | remains logical and persistence-neutral; cardinalities and immutable-revision/provenance relationships must track public contracts and must not invent a hosted DB |
| Requirements traceability | `docs/traceability/requirements-matrix.md` | PARTIAL | useful baseline exists; refresh after material protected merges and active-PR state changes so no active work is presented as shipped |
| Scientific / standards basis | `docs/traceability/research-basis.md` + doctoring | PRESENT_CURRENT | use APA 7, primary sources, stable links, scope/equation traceability, and conservative interpretation boundaries |
| Reusable-core threat model | `docs/security/threat-model.md` | PRESENT_CURRENT | update on native/provider/artifact/serialization/secret/resource/trust-boundary changes |
| Public interface/version/serialization/fingerprint contract | indexed public-contract docs + ADR-0003/0011 | PARTIAL | make canonicalization, schema/version, compatibility/deprecation, fingerprint preimage, cross-language vectors, and cross-repository handoffs explicitly discoverable |
| Reusable-core operability/recovery | resource/failure/release docs + TRD/V&V | PARTIAL | package-owned bounded resources, cancellation, deterministic failure evidence, fallback/degraded behavior, wheel/reinstall recovery, and diagnostics need one discoverable operating index |
| Security/data-governance index | threat model + ADR-0012 + security docs | PARTIAL | purpose limitation, authorization, retention/export responsibility, provider/secret/raw-content boundaries, supply-chain ownership, and responsible disclosure must stay discoverable |
| Release/migration/rollback/provenance/licensing index | release acceptance + packaging/SBOM/provenance/licensing docs | PARTIAL | existing controls are substantial but need one canonical index across compatibility, rollback, artifact hashes, SBOM/provenance, reproducibility, NOTICE/licensing, and release acceptance |
| Documentation contract CI | architecture/documentation contract tests | PRESENT_CURRENT | documentation-as-code checks are protected-main behavior; extend them when status vocabulary, canonical inventory, links, UML/ERD source hygiene, or ownership rules change |
| Root README / AGENTS / CLAUDE / CHANGELOG alignment | root authority files | PARTIAL | fail documentation fitness when obsolete product names, authoritative NumPy-first claims, stale version support, or contradictions with PRD/TRD/Architecture reappear |
| Hosted product operational runbook | Psychometrics Commons/operator docs | NOT_APPLICABLE | hosted tenant/session/consent/database/UI/deployment operations stay downstream |
| Physical product DB schema/migrations | Psychometrics Commons/owning host | NOT_APPLICABLE | do not manufacture ORM/DDL merely to satisfy an ERD request |
| Tenant/RBAC/SSO/SCIM/UI/billing | hosted product/services | NOT_APPLICABLE | retain only explicit versioned reusable-core handoff boundaries |

## Protected-main scientific and product maturity

The table below records product truth, not documentation-file presence. “Implemented” means ancestral to protected main, not merely discussed in an issue or present on an open branch.

| Work family | Current maturity | Evidence / remaining boundary |
|---|---|---|
| Fallible human/AI/LLM raters and many-facet scoring | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | governed scoring/facet primitives exist; broader discrimination/range/drift extensions remain incremental |
| Correlation is not recovery/agreement/validity | IMPLEMENTED_ON_PROTECTED_MAIN | recovery and interpretation governance is protected-main policy |
| Reference-free RAG request/provenance boundary | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | governed privacy-preserving RAG scoring-request adapter is integrated; full calibration/validation/bank workflow remains broader work |
| Dynamic evidence-grounded rubric generation | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | rubric/blueprint/generation/audit/pilot primitives exist; closed-loop governed bank evolution remains broader than generation |
| Governed post-pilot item-bank lifecycle | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | immutable post-pilot lifecycle/evidence gates are integrated; linking/exposure/drift/assembly/release integration continues incrementally |
| Bifactor / higher-order / testlet / two-tier / many-facet relation governance | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | relation-safe policy is established; family-specific estimator/validation evidence varies |
| Latent-space residual interaction | IMPLEMENTED_ON_PROTECTED_MAIN | interpretation remains gated on substantive dimension/testlet/facet diagnosis |
| Angoff delta-plot observed-score DIF | IMPLEMENTED_ON_PROTECTED_MAIN | `fast_mlsirm.delta_plot`; method page and ADR-0018; flags are screens, not fairness determinations |
| Bradley–Terry MM pairwise ranking | IMPLEMENTED_ON_PROTECTED_MAIN | `bradley_terry_mm` and additive-ties `bratt_mm`; ADR-0017; Rao–Kupper/Davidson remain unimplemented |
| Formal non-nested distinguishability/model comparison | PARTIAL | fail-closed relation-aware comparison exists; additional family-specific evidence and metadata remain incremental |
| Adaptive rotation criterion selection | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | Rust-backed criterion registry/multi-start selector/report surfaces are integrated; additional criteria/GPU/recovery remain incremental |
| Multilevel / cross-classified / multiple-membership contracts | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | contextual and longitudinal contracts are integrated; crossed / multiple-membership `u_h` MAP estimation with RMSE recovery is this kernel; OLS/AR and richer variance-component claims remain separate |
| Temporal/longitudinal/drift estimators | PARTIAL | governed contracts/design primitives exist; continuous-time or richer estimator claims require separate recovery evidence |
| Automated essay scoring calibration/validation | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | governed essay contracts/validation/reporting exist; generalized rater discrimination/range/drift remains incremental |
| Paired automated-vs-reference rating-range evidence | IMPLEMENTED_ON_PROTECTED_MAIN | Rust-owned paired range/compression diagnostic is integrated |
| Enterprise issue measurement | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | reusable evidence/calibration adapters exist; causal intervention utility remains downstream/policy-bound |
| Factor retention evidence contract | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | governed factor-retention evidence contract is integrated; structural model selection remains a distinct subsequent decision |
| Rust-first numerical ownership | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | fixed-anchor linking, CAT/ATA, covariance, observed-information Hessian assembly, second-order diagnostics, and JMLE Adam/L-BFGS optimizer sequencing are protected-main Rust/PyO3 paths; remaining kernel-specific migrations stay explicit |
| Fixed-anchor parameter linking arithmetic | IMPLEMENTED_ON_PROTECTED_MAIN | protected main owns scale/shift estimation and theta/alpha/b transformation in Rust/PyO3 |
| Observed-information Hessian and second-order diagnostics | IMPLEMENTED_ON_PROTECTED_MAIN | protected main owns finite-difference coefficients/symmetric Hessian assembly and eigenvalue/positive-definiteness diagnostics in Rust/PyO3; Python only evaluates objective samples and transports results |
| JMLE Adam/L-BFGS optimizer arithmetic | IMPLEMENTED_ON_PROTECTED_MAIN | PR #760 is ancestral to current protected main; `backend="rust"` delegates Adam/L-BFGS/combined optimizer control to compiled Rust while recovery evidence remains governed separately by issue #626 |
| Parallel-analysis public control/resource hardening | IMPLEMENTED_ON_PROTECTED_MAIN | strict integer/control validation and bounded random-benchmark workspace ceilings are ancestral to protected main |
| Hourly review-repair caller | IMPLEMENTED_ON_PROTECTED_MAIN / PARTIAL | PR #763 integrated the product-side bounded caller; operational scheduler/control-plane acceptance remains external evidence rather than a library capability |
| LLM-judge raw JSON depth hardening | IMPLEMENTED_ON_PROTECTED_MAIN | PR #764 is ancestral to current protected main and bounds recursive JSON nesting before parser materialization |
| Essay-report native dark-mode status accents | IMPLEMENTED_ON_PROTECTED_MAIN | CSS-variable and prefers-color-scheme dark-mode status accents are ancestral to protected main |
| Canonical PyO3/public-export governance | ACCEPTED_ARCHITECTURE / PARTIAL | ADR-0011 governs convergence; feature-by-feature hardening continues |
| Purpose-limited sensitive-data handling | IMPLEMENTED_ON_PROTECTED_MAIN / DOWNSTREAM | reusable contracts prefer purpose limitation/minimization/separated identities; hosted authorization/retention execution remains downstream |
| LLM orchestration/model credentials | IMPLEMENTED_ON_PROTECTED_MAIN | provider execution and independent reviewer identity/credential boundaries are governed; provider calls remain outside psychometric numerical core |
| Continuous execution and canonical documentation ownership | IMPLEMENTED_ON_PROTECTED_MAIN | ADR-0013 is integrated documentation/process authority; runtime scheduler state is external and is not a shipped library capability |

## Current active-PR boundary

At this review, material open work includes:

- Rust allocation preflight for parallel analysis, if it remains separate;
- the documentation-fitness refresh itself, which may describe current protected truth but is not authoritative until merged.

These remain active-PR evidence, not protected-main capability. Their source heads, checks, reviews, writer leases, and mergeability are operational evidence and must be re-fetched rather than copied into timeless architecture prose.

## P0 documentation gaps

A P0 documentation defect blocks calling the affected architecture/release story complete when any of the following is true:

- missing or materially stale canonical PRD, TRD, root architecture, ADR authority, standards registry, V&V/Test Strategy, threat model, logical data/evidence model, or requirement traceability;
- an active PR, issue, target diagram, research result, or scheduler behavior is promoted to protected-main product truth;
- public interface/version/serialization/fingerprint behavior cannot be reconstructed without source archaeology;
- UML/ERD names, ownership, state, cardinality, or conceptual-vs-persisted semantics contradict the public contract;
- an obsolete early-MVP/NumPy-first/product-name/version claim competes with the current Rust-first protected-main authority;
- hosted product database/HTTP/tenant/RBAC ownership is moved into `fast-mlsirm` without a superseding accepted decision; or
- release, migration, rollback, provenance, licensing, operability, or security ownership for the affected capability cannot be discovered from the canonical graph.

## P1 documentation gaps

P1 gaps do not automatically block unrelated development, but must be closed before releasing or making the affected claim:

- missing method-specific doctoring or primary-source traceability;
- missing recovery/scoreability/identification interpretation boundary;
- missing resampling/generalization unit in V&V;
- missing failure/recovery/rollback instructions for a changed public artifact;
- missing privacy/security abuse case for a new provider/native/artifact surface;
- missing changelog/release/provenance evidence for an accepted user-visible capability; or
- missing machine-check for a high-risk documentation invariant that has already drifted in practice.

## P2 improvements

- richer rendered architecture/site navigation generated from the canonical graph;
- stronger PlantUML renderability/link checking in normal CI where supported;
- generated traceability views from contract metadata;
- downstream buyer/operator views that consume, rather than duplicate, canonical artifacts; and
- optional downstream Figma/UX links for hosted workbench experiences.

## Maintenance gate

Every material PR should answer:

1. Did product requirements, users/JTBD, or non-goals change?
2. Did a technical invariant, public contract, trust/resource/privacy/release rule, or ownership boundary change?
3. Did a durable architecture/scientific decision change or require supersession?
4. Did an applicable published standard edition/status or watch item change?
5. Did software/numerical/scientific V&V evidence, identification, recovery, scoreability, or generalization unit change?
6. Did component/data/lifecycle/deployment/authority/recovery/UML/ERD views change?
7. Did the threat model or data-governance boundary gain an asset, actor, abuse case, provider, secret, native boundary, or retention/export responsibility?
8. Did capability maturity change between planned, active-PR, protected-main, downstream, or superseded states?
9. Did source/test/evidence move enough that requirements/research traceability is stale?
10. Are public interface/version/serialization/fingerprint compatibility and deprecation rules still discoverable?
11. Are operability, migration/rollback, SBOM/provenance/reproducibility/licensing and release evidence synchronized?
12. Do README, AGENTS, CLAUDE, Architecture, PRD/TRD and CHANGELOG tell one consistent current story?

If any answer is yes, update the corresponding canonical artifact in the same coherent workstream or record a precise `NOT_APPLICABLE`/downstream/no-change justification. Documentation drift is a repository defect, not post-release cleanup.
