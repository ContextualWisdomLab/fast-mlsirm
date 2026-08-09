# Capability Maturity and Ownership Map

Audit baseline: protected `main` `4d910ed650f384ff882c8b5fba6a8b08fd532236` on 2026-08-09. This document prevents target architecture from being mistaken for shipped capability. Refresh the baseline commit and classifications whenever a material feature reaches protected main.

## Status vocabulary

- **IMPLEMENTED** — an accepted protected-main public/runtime path exists with repository tests/evidence.
- **PARTIAL** — meaningful protected-main capability exists, but the full architecture/claim is not yet implemented or identified.
- **DESIGN-REQUIREMENT** — accepted product/technical constraint for future implementation; must not be advertised as available.
- **OWNED-BY-OTHER-REPO** — explicitly outside the `fast-mlsirm` bounded context.
- **NOT-APPLICABLE** — deliberately absent from this repository.

Documentation, open pull requests and plans cannot promote a capability above the state established by protected-main implementation and accepted evidence.

## Numerical and psychometric core

| Capability | Status | Protected-main boundary / interpretation |
|---|---|---|
| Rust/PyO3 primary numerical backend | IMPLEMENTED | Compiled Rust core is the primary numeric path; Python/NumPy remains reference/fallback where explicitly supported and parity-tested. |
| Simple-structure MLSIRM/MLS2PLM simulation and fitting | IMPLEMENTED | Current formula scope is the simple-structure specialization documented in `AGENTS.md`; full discrimination-vector MLS2PLM is a separate model path if developed. |
| MIRT/MLSRM/MLS2PLM/ULSRM/ULS2PLM constraints | IMPLEMENTED | Public fitting/simulation/diagnostic behavior remains bounded by current source and tests. |
| Polytomous/testlet/facet/linking/CAT/ATA methods | IMPLEMENTED/PARTIAL | Multiple method families are present; each method's doctoring/tests define its actual support and interpretation limits. The architecture baseline is not evidence that every family has identical estimation depth. |
| Bifactor scoreability | PARTIAL | Scoreability/index APIs exist or are being integrated by method-specific work; improved fit never authorizes score interpretation without applicable evidence. |
| Adaptive rotation criterion selection | PARTIAL | Rotation machinery/criteria exist, while criterion-neutral automated selection and exhaustive recovery evidence remain feature-specific. Finite multi-start is never called a global optimum. |
| Relation-safe structural model selection | PARTIAL | Base comparison functionality exists; complete relation classification, formal distinguishability and boundary-aware public workflows remain gated by feature-specific implementation/evidence. |
| GPU acceleration | IMPLEMENTED/PARTIAL | GPU paths exist for material kernels with parity/no-skip requirements. No claim is made that every numerical method is GPU-accelerated. |
| Low-contention CPU multithreading | IMPLEMENTED/PARTIAL | Used on selected Rust hot paths; it is an architecture requirement for new material kernels, not a claim that every operation is parallel. |

## Hierarchy, multiple membership and time

| Capability | Status | Boundary |
|---|---|---|
| Group/cluster-aware diagnostics and context summaries | IMPLEMENTED | Existing diagnostics may consume group/cluster context according to their method contracts. |
| Explicit nested/cross-classified/multiple-membership contracts | PARTIAL / DESIGN-REQUIREMENT | Representation and scientific rules are accepted; do not claim a general production multilevel estimator until its exact Rust estimator, identification and recovery evidence land. |
| General multilevel/multiple-membership psychometric estimator | DESIGN-REQUIREMENT | Requires connectedness/identification, realistic true-parameter bias/RMSE/coverage and backend evidence before support claims. |
| Discrete longitudinal/occasion semantics | PARTIAL / DESIGN-REQUIREMENT | Ordering/occasion provenance is required where used. Feature-specific implementations define available dynamics. |
| Continuous-time psychometric transition model | DESIGN-REQUIREMENT | Not implied by timestamps or discrete AR parameters; requires separately identified units, likelihood and recovery. |
| Broad event/relationship/trajectory analytics | OWNED-BY-OTHER-REPO | TEPP owns broader temporal/event analytical artifacts; `fast-mlsirm` exchanges versioned measurement artifacts only. |

## Rubric, scoring and AI evaluation

| Capability | Status | Boundary |
|---|---|---|
| RubricSpecification / AssessmentSpec / scoring contracts | IMPLEMENTED/PARTIAL | Reusable canonical contracts exist and continue to expand through versioned compatible changes. |
| Bounded rubric/blueprint/generation contracts | IMPLEMENTED/PARTIAL | Structured provider-neutral generation contracts exist; generated content remains untrusted until screening/calibration. |
| Full governed item-bank lifecycle | PARTIAL / DESIGN-REQUIREMENT | The lifecycle in ADR-0009 is normative design. Individual draft→screen→pilot→calibrated→approved→active→quarantine/retire transitions may not all have a protected-main runtime implementation yet. |
| Automated essay scoring evidence/reporting | IMPLEMENTED/PARTIAL | Criterion-level scoring/validation/reporting exists. Additional rater range/drift/fairness evidence remains feature-specific. |
| Human/LLM rater calibration principle | IMPLEMENTED/PARTIAL | Shared fallible-rater contracts and many-facet capabilities exist; no human or model is treated as infallible truth. |
| Reference-free RAG canonical observation schema | DESIGN-REQUIREMENT | Research/design direction is documented; do not advertise a stable public RAG schema until accepted into the package. |
| Enterprise issue evidence/measurement adapters | IMPLEMENTED/PARTIAL | Measurement-oriented enterprise issue work exists; consequential intervention prioritization remains a separately identified downstream decision layer. |
| Provider-specific LLM gateway | OWNED-BY-OTHER-REPO | `contextual-orchestrator` owns general provider routing/orchestration; `fast-mlsirm` owns provider-neutral measurement contracts. |
| External network/SSRF policy engine | OWNED-BY-OTHER-REPO | EgressWeave or the host owns outbound authority controls. |

## Product/runtime/persistence boundary

| Capability | Status | Boundary |
|---|---|---|
| Local Python/Rust package, CLI and deterministic reports | IMPLEMENTED | Core standalone product surface. |
| Release acceptance, benchmark, buyer/procurement/provenance evidence | IMPLEMENTED | Repository scripts provide exact-artifact evidence; each release must still pass current gates. |
| Physical application database / ORM | NOT-APPLICABLE | `docs/ERD.md` is logical and persistence-neutral only. |
| Hosted participant/session/response/consent/result lifecycle | OWNED-BY-OTHER-REPO | Psychometrics Commons. |
| Hosted tenant/resource authorization and product migrations | OWNED-BY-OTHER-REPO | Psychometrics Commons and identity/security bounded contexts. |
| Identity/federation/passkeys | OWNED-BY-OTHER-REPO | Keyverse. |
| Hosted UI/workbench/reference clients | OWNED-BY-OTHER-REPO | Psychometrics Commons or the relevant client repository; Figma work belongs with stable product flows. |
| Organization-wide CI/control plane | OWNED-BY-OTHER-REPO | `ContextualWisdomLab/.github`. |

## Governance/compliance maturity

| Capability | Status | Boundary |
|---|---|---|
| Software quality / scientific evidence gates | IMPLEMENTED/PARTIAL | CI, security, package, provenance and scientific-recovery gates exist and evolve; exact current branch protection/workflows are authoritative. |
| WCAG-aligned deterministic report accessibility | IMPLEMENTED/PARTIAL | Implemented report surfaces have accessibility tests/doctoring; this does not claim every future hosted UI is conformant. |
| AI risk/impact governance guidance | DESIGN-REQUIREMENT | ISO/IEC 42001, ISO/IEC 23894, ISO/IEC 42005 and NIST AI RMF family inform controls; this repository does not claim certification. |
| Formal CSAP/SOC 2 certification/control matrix | OWNED-BY-OTHER-REPO / DESIGN-REQUIREMENT | Core may supply reusable evidence, but formal hosted/control-plane ownership and audit evidence belong primarily to Psychometrics Commons and `.github`/infrastructure owners. |

## Promotion rule

A row may be promoted to **IMPLEMENTED** only after the accepted protected-main path includes the relevant public contract, implementation, realistic tests, required scientific/recovery evidence, documentation, security/privacy controls, packaging/release evidence and current-head review. A design document or merged interface stub alone is insufficient.
