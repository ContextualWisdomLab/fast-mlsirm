# Requirements and evidence traceability matrix

Status: **Authoritative gap/coverage map**  
Last reconciled: **2026-08-09**

This matrix consolidates the major accepted work families developed across the project conversation and repository history. `Implemented` means protected `main` contains the named production boundary; it does **not** mean every future extension in that family is complete. `Active` means work exists on an open PR and cannot be counted as released. `Planned` means the architecture is accepted/proposed but the full product path is not yet on protected `main`.

| Work family | PRD | TRD | ADR | Current code / API authority | Test / evidence authority | Status / remaining gap |
|---|---|---|---|---|---|---|
| Repository ownership: reusable core vs hosted product | PRD-014 | TRD-024 | ADR-0001 | `AGENTS.md`, `CLAUDE.md` | documentation boundary tests | Implemented; keep hosted persistence/UI in Psychometrics Commons |
| Rust numerical source of truth, Python orchestration, GPU parity | PRD-005, 013 | TRD-001, 002, 006-008 | ADR-0002 | `crates/mlsirm-core`, `crates/fast-mlsirm-py`, package root | Rust/PyO3/package/GPU tests | Implemented architecture; canonical binding registry must remain conflict-free as features grow |
| AssessmentSpec / scoring-policy contracts | PRD-001 | TRD-003, 005 | ADR-0003 | `python/fast_mlsirm/scoring/assessment.py` and scoring contracts | assessment/scoring contract tests | Implemented |
| RubricSpecification and blueprint compiler | PRD-001, 002 | TRD-003, 004 | ADR-0003, 0004 | `python/fast_mlsirm/rubric/models.py`, `compiler.py` | rubric authoring/version/blueprint tests | Implemented |
| Provider generation request and hostile candidate validation | PRD-003 | TRD-004, 017 | ADR-0004 | `python/fast_mlsirm/rubric/generation.py` | `test_rubric_generation*`, provenance tests | Implemented structural/provenance boundary; semantic/psychometric screening remains incomplete as one closed loop |
| Governed semantic screening and item-bank lifecycle | PRD-002, 012 | TRD-004, 023 | ADR-0004 | feature-specific screening primitives; no hosted bank persistence | future answerability/alignment/leakage/fairness/calibration lifecycle evidence | Planned/partial; largest upstream buyer-facing gap |
| Human/LLM/external scorer common observations | PRD-004, 011 | TRD-005, 015 | ADR-0003, 0005 | `python/fast_mlsirm/scoring/` including execution/authorization and domain adapters | scoring execution, replay, rater evidence tests | Implemented reusable contract layer |
| Automated essay scoring validation/calibration | PRD-011, 016 | TRD-005, 015, 028 | ADR-0005 | `python/fast_mlsirm/scoring/essay/` and shared facets reporting | essay validation/reporting tests | Implemented initial calibration/validation/report stack; inferential rater range/discrimination/drift extensions continue |
| Enterprise issue evidence -> criterion observation -> facets reports | PRD-004, 011 | TRD-005, 015, 028 | ADR-0003, 0005 | `fast_mlsirm.scoring.enterprise_issue` shared-contract adapters | enterprise provenance/calibration/report tests | Implemented measurement layer; causal net-intervention-value decision policy remains downstream/non-goal |
| Reference-free RAG as measurement | PRD-010 | TRD-005, 009-015 | ADR-0005 | shared Assessment/Rubric/Scoring/Facets/Testlet/MIRT/LSIRM primitives | current psychometric and rater tests | Architecture supported; canonical RAG observation adapter/data contract and end-to-end benchmark work remain Planned |
| Many-facet rater severity / connectedness | PRD-004, 005, 011 | TRD-015 | ADR-0005 | `fit_facets` and scoring calibration/report helpers | facets/connectedness/agreement tests | Implemented baseline; rater discrimination/range/drift models require their own recovery evidence |
| Correlated multidimensional / MLSIRM model family | PRD-005, 006 | TRD-006, 008-010 | ADR-0002, 0005 | package estimator/model APIs + Rust core | likelihood/gradient/recovery/fit tests | Implemented supported families per current package exports |
| Bifactor model structure and scoreability | PRD-006 | TRD-011 | ADR-0005 | BIFAC-related model paths and bifactor diagnostic APIs as released | bifactor index/identity/recovery evidence | Implemented in current released scope; categorical reliability and new structures must not be inferred from latent-response indices |
| Higher-order / two-tier structural comparison | PRD-006 | TRD-009, 010 | ADR-0005 | existing multidimensional/CDM primitives; method-specific support varies | model/recovery tests | Partial; formal relation determined by actual parameter constraints, not model name |
| Relation-safe nested/non-nested comparison | PRD-007 | TRD-009, 010 | ADR-0005 | model comparison APIs and Rust Vuong kernel | comparison/cluster/bootstrap tests | Implemented fail-closed wrapper; complete formal distinguishability support remains an active scientific extension where score/information inputs are missing |
| Factor retention | PRD-008 | TRD-009, 010, 012 | ADR-0005, 0007 | dimensionality/fit utilities | fit/CV/recovery evidence | Partial/expanding; no single retention statistic is authoritative |
| Adaptive rotation criterion registry/selector | PRD-008 | TRD-012 | ADR-0007 | rotation APIs where integrated | gradient/covariance/multi-start/bootstrap/recovery tests | Architecture accepted; feature support is version-specific and active work must pass integration/PyO3 evidence before release claims |
| Testlet/local-dependence handling | PRD-006, 009 | TRD-010, 013 | ADR-0005, 0006 | `fit_testlet`, fit/local-dependence diagnostics | testlet/Q3/local dependence/recovery tests | Implemented baseline; full bifactor/testlet/latent-space combinations require evidence before adoption |
| Multilevel/cross-classified/multiple-membership contracts | PRD-009 | TRD-013 | ADR-0006 | dedicated governed namespace is active PR work; existing multilevel-adjacent support remains | branch recovery/design tests when integrated | Active, not released from this documentation baseline |
| Temporal/longitudinal/drift structure | PRD-009, 011 | TRD-013, 015 | ADR-0006 | current occasion/rater monitoring primitives; dedicated longitudinal contract work active | drift/order/recovery tests | Partial/Active; discrete-step AR parameters must not be called continuous-time dynamics |
| G-theory design reliability | PRD-005, 013 | TRD-014 | ADR-0005, 0008 | `gtheory_pio` / related utilities | G-study/D-study tests | Implemented where exposed |
| True-parameter recovery over correlation | PRD-013 | TRD-014 | ADR-0008 | recovery/simulation APIs and scientific study scripts | bias/RMSE/coverage/convergence studies | Implemented governance; continue expanding model-specific recovery |
| Heavy scientific studies vs PR smoke | PRD-013, 017 | TRD-018, 025, 027 | ADR-0008 | statistical study workflows/scripts | shard inventory/deadline/recovery workflow tests | Implemented direction; active reliability work continues for bounded subprocess/process-tree handling |
| Accessible deterministic reports | PRD-016 | TRD-016, 028 | ADR-0005, 0008 | report modules including essay validation HTML/JSON | accessibility/render/provenance tests | Implemented and actively refined |
| PII/purpose-limited audit boundary | PRD-015 | TRD-020-022 | ADR-0001, 0003, 0004 | source-free/digest-based governed contracts where possible | privacy/provenance/adversarial metadata tests | Implemented reusable boundary; downstream identity/retention/residency belongs to hosted owner |
| CSAP/SOC 2 readiness without certification claim | PRD-015, 017 | TRD-020, 027 | ADR-0001, 0008 | CI/security/provenance/docs controls | Security Scan/SAST/SBOM/release evidence | Ongoing control-evidence target, not certification |
| NVIDIA NIM autonomous/model-backed path; no Copilot dev token | PRD-015 | TRD-021 | ADR-0008 | organization-controlled automation integrations | workflow contract/security tests | Governed by repository/central automation; reviewer credentials remain independent |
| Release / SBOM / provenance / exact-head evidence | PRD-017 | TRD-020, 025, 027, 028 | ADR-0008 | release workflows, changelog renderer, package metadata | CI/security/package/release acceptance | Implemented governance; release only when exact artifact evidence is complete |
| UML/C4/logical ERD/traceability as release docs | PRD-017 | TRD-026 | ADR index | `ARCHITECTURE.md`, this matrix, `docs/architecture/diagrams.md` | `tests/test_architecture_documentation_contract.py` | Added by this baseline; must remain synchronized |

## Traceability gaps that remain executable work

### A. Governed item-bank closed loop

Protected `main` has strong rubric, generation, scoring, calibration, and reporting primitives, but the buyer-visible lifecycle from semantic screening through pilot/calibration/approval/linking/drift/retirement is not yet one cohesive reusable workflow. This is a product gap, not a documentation-only gap.

### B. Canonical reference-free RAG adapter

The conversation establishes a clear measurement model, but the repository still needs a canonical domain-neutral observation contract that binds question/evidence/response/atomic criterion/judge/prompt/occasion/system-run/query-testlet identities into the shared scoring/calibration path.

### C. Full rater-effect extensions

Severity/connectedness are established. Rater discrimination, range restriction, criterion-specific bias, and longitudinal drift require separate identified models and true-parameter recovery before they can become generalized product claims.

### D. Multilevel and temporal release integration

The dedicated contextual/longitudinal contract work is active and must not be marked released until protected-main integration and exact-head scientific evidence complete.

### E. Formal distinguishability inputs

Relation-safe comparison already refuses unsupported conclusions. Complete formal Vuong distinguishability requires model families to expose the casewise score/information metadata required by the method; until then the product must continue to return an explicit “requires distinguishability evidence” state.

### F. Buyer workbench

`fast-mlsirm` should define reusable workflow/evidence contracts, but the hosted visual workflow (Rubric authoring -> Blueprint -> Candidate review -> Calibration -> Bank approval) primarily belongs downstream in Psychometrics Commons. Figma/Product Design should therefore be driven by stable contracts and handed off rather than recreated as hosted UI here.

## Update rule

A material PR that changes one of these work families must update this row (or add a row) in the same change. A row marked `Active` cannot be promoted to `Implemented` until the relevant code is on protected `main` and its exact-head acceptance evidence is available.
