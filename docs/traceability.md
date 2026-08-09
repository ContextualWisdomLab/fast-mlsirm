# fast-mlsirm Research, Requirement, and Implementation Traceability

**Snapshot:** protected `main` `4d910ed650f384ff882c8b5fba6a8b08fd532236` plus open PR state reviewed on 2026-08-09.  
**Purpose:** make the major product/research decisions developed across project discussions discoverable without reconstructing chat history.

Status vocabulary:

- `implemented_on_main` — source/docs/tests found on the protected-main snapshot.
- `open_pr` — implementation exists only on an unmerged PR at the snapshot.
- `planned` — accepted product direction, not proven integrated.
- `research_only` — research conclusion/hypothesis without a public product contract.
- `out_of_scope` — owned by another bounded context.

## 1. Product and architecture traceability

| Topic / requirement | Status | Protected-main evidence or owner | Authoritative decision/doc |
|---|---|---|---|
| Rust-first psychometric numerical authority | implemented_on_main | `crates/mlsirm-core/`, `crates/fast-mlsirm-py/`, parity/package tests | `ARCHITECTURE.md`, ADR-0002 |
| Independent reusable measurement library | implemented_on_main | package layout; `AGENTS.md`; `CLAUDE.md` | ADR-0001 |
| Hosted assessment/runtime product | out_of_scope | `ContextualWisdomLab/psychometrics-commons` | ADR-0001 |
| Canonical assessment/scoring contracts | implemented_on_main | `python/fast_mlsirm/scoring/assessment.py`, `scoring/contracts.py`, scoring contract tests | PRD §3.3; ADR-0003 |
| Rubric specification and blueprint compiler | implemented_on_main | `python/fast_mlsirm/rubric/models.py`, `rubric/compiler.py`, `rubric/contracts.py` | ADR-0003; `docs/rubric_item_generation.md` |
| Generated provider-output validation | implemented_on_main | `python/fast_mlsirm/rubric/generation.py`; `docs/rubric_generation_validation.md` | ADR-0003 |
| Full governed item-bank lifecycle | planned | no canonical bank lifecycle API located on snapshot | PRD §3.4; UML state diagram; ERD |
| Automated scoring common engine | implemented_on_main | `python/fast_mlsirm/scoring/`, execution/authorization/migrations | PRD §3.5 |
| Automated essay scoring adapters/contracts | implemented_on_main | `python/fast_mlsirm/scoring/essay/`; essay docs/tests | PRD §3.5 |
| Automated essay validation evidence reports | implemented_on_main | essay validation/reporting modules and HTML/JSON tests | TRD §7; existing essay doctoring |
| Enterprise issue scoring/calibration contracts | implemented_on_main | `python/fast_mlsirm/scoring/enterprise_issue/`; enterprise calibration handoff | PRD §3.7 |
| Enterprise causal intervention utility | out_of_scope / planned downstream | not a psychometric calibration parameter | PRD §3.7; ADR-0001 |
| Reference-free RAG measurement contract | planned | no dedicated RAG measurement namespace found on snapshot | PRD §3.6; ADR-0003 |
| RAGAS adapter | planned | no dedicated adapter found on snapshot | PRD §3.6 |
| Candidate-blind/cross-fitted dynamic rubric policy | planned / research-grounded | rubric infrastructure exists; benchmark-specific enforcement not located as a full product path | ADR-0003 |
| Psychometric CI/CD as full merge gate | planned / partial | extensive recovery/security/package gates exist; unified psychometric CI policy not yet one product API | PRD, TRD §10 |
| Figma buyer workbench | out_of_scope here / downstream product | UI belongs downstream | ADR-0001 |

## 2. Measurement-method traceability

| Method / scientific requirement | Status | Evidence / gap | Governing rule |
|---|---|---|---|
| MLSIRM / MLS2PLM simple-structure production path | implemented_on_main | existing Rust/Python objective, simulation, fit, recovery | `AGENTS.md`; ADR-0002 |
| Full discrimination-vector MLS2PLM | planned model-design change | current formula intentionally simple-structure | `AGENTS.md` formula scope |
| MIRT / related constrained models | implemented_on_main | public model/fit infrastructure | `ARCHITECTURE.md` |
| Polytomous scoring/model infrastructure | implemented_on_main | Rust scoring/poly modules/tests and public package capabilities | PRD §3.1 |
| Many-facet/rater calibration | implemented_on_main | scoring/facet workflows and reports | PRD §3.5; ADR-0003 |
| Bifactor scoreability | implemented_on_main | `python/fast_mlsirm/bifactor_scoreability.py`, docs/tests | ADR-0004 |
| Formal higher-order-vs-bifactor relation derived from constraints | planned enhancement | current generic model-comparison API exists; structural relation inference remains a broader requirement | ADR-0004 |
| Non-nested model comparison API | implemented_on_main | `python/fast_mlsirm/model_comparison.py`, tests/docs | ADR-0004 |
| Formal Vuong distinguishability end-to-end for all fit families | planned | comparison API deliberately fails closed before unsupported formal inference | ADR-0004; TRD §5.2 |
| Adaptive factor rotation / criterion selection | implemented_on_main | `rotation_selection.py`, Rust rotation selector, PyO3 bindings, docs/tests | ADR-0004; TRD §9.2 |
| Comprehensive factor-retention workflow | partial / planned | parallel/factor utilities exist; one authoritative retention workflow across response types remains a gap | TRD §9.1 |
| Testlet/local-dependence handling | implemented_on_main / evolving | testlet and local-dependence diagnostics exist; rubric-testlet pilot handoff exists | ADR-0004 |
| Two-tier model | planned unless source-specific implementation is proven | architecture requires it for multiple primary + secondary dimensions | ADR-0004 |
| Multilevel/cross-classified/multiple-membership reusable contracts | open_pr | PR #566 (`feat(multilevel): add contextual and longitudinal contracts`) | ADR-0004; TRD §6 |
| Continuous-time likelihood | planned / research_only | PR #566 explicitly limits current concept to discrete occasion ordering/AR(1) contract | ADR-0004 |
| Latent-space residual interaction | implemented_on_main for existing MLSIRM paths | do not use as substitute for omitted factor/facet/testlet structure | ADR-0004 |
| True-parameter recovery > correlation-only validation | implemented_on_main as project rule; evolving coverage | recovery tests and research docs | ADR-0002; TRD §10 |
| CPU/GPU parity where GPU claimed | implemented_on_main for existing GPU paths; method-specific gaps remain | GPU tests and explicit no-skip requirements | ADR-0002 |

## 3. RAG / LLM-as-a-Judge research conclusions

The project discussions established the following accepted research constraints even where the final product namespace is not yet integrated:

1. RAGAS and LLM-judge results are observations with measurement error, not truth.
2. `reference-free` is not `truth-free`; groundedness to retrieved context differs from world correctness and completeness.
3. Retrieval relevance and context utilization are distinct constructs.
4. Claim-level binary/binomial, ordinal rubric, continuous, and pairwise observations should use response models appropriate to their measurement scale rather than arbitrary thresholding.
5. Judge severity, discrimination, prompt/order/occasion, and family dependence can affect evaluation and require calibration/sensitivity analysis.
6. Common/query-specific criteria require a testlet or other local-dependence strategy when they share the same query/context/answer.
7. A general RAG quality score should be justified by correlated-MIRT/bifactor/second-order comparison and scoreability evidence, not assumed.
8. Latent space is a residual interaction layer after substantive dimensions/facets/testlets.
9. Candidate-aware rubric discovery must be separated from final scoring for fair benchmarking.
10. Human audit, when available, is a calibration/validity source rather than an assumption that one raw human label is error-free.

**Product status:** research-grounded direction; reusable rubric/scoring infrastructure is integrated, but a complete canonical reference-free RAG measurement pipeline remains `planned`.

## 4. Automated-scoring research conclusions

Accepted constraints represented in current requirements:

- human and AI ratings share a rater-observation framework;
- correlation alone is association, not agreement/accuracy/validity;
- use QWK/exact/adjacent agreement for ordinal agreement and MAE/RMSE/calibration when a defensible criterion exists;
- estimate or diagnose rater severity and, where identified, criterion bias, discrimination/consistency, range restriction, and drift;
- inspect DIF/invariance and subgroup error;
- preserve rubric/model/prompt versions and evidence;
- route uncertain/disagreeing/high-risk cases for human review rather than forcing an automatic decision;
- separate measurement evidence from the downstream consequential-decision policy.

## 5. Dynamic rubric and item-generation research conclusions

The intended closed loop is:

```text
Rubric
→ measurement blueprint
→ bounded candidate generation
→ structural/evidence/semantic screening
→ artificial crowd / human+AI pilot
→ item/rater calibration
→ information/content-constrained assembly
→ immutable bank version
→ drift/DIF/exposure monitoring
→ rubric revision
```

Protected main contains rubric specification/compiler/generation-validation primitives, but the fully governed item-bank lifecycle and complete artificial-crowd orchestration remain product gaps.

## 6. Enterprise issue measurement conclusions

The reusable measurement layer may estimate constructs such as materiality, urgency evidence, actionability evidence, rater/facet behavior, and stakeholder disagreement. Final organizational priority is not a raw latent score. Where a downstream decision product exists, expected net intervention value conceptually depends on counterfactual outcome benefit and intervention cost; these causal/business-utility components are not silently encoded into IRT item discrimination.

## 7. Security/compliance traceability

| Requirement | Status / boundary |
|---|---|
| SOC 2 / CSAP readiness | design toward controls; no certification claim |
| PII handling | purpose limitation/minimization and identity separation; no blanket masking requirement in the library |
| Identity/tenant authorization | downstream hosted owner; library keeps reusable authorization/evidence primitives only where domain-neutral |
| Central Security Scan/SAST/fuzz/package | repository gates; must remain fail-closed |
| LLM development key | `NVIDIA_NIM_API_KEY` for genuine model-backed workflow; do not use `COPILOT_GITHUB_TOKEN` |
| Review/writer separation | independent reviewer credentials/identity; no manufactured approval |
| Self-modifying privileged CI | prohibited final-state architecture |
| Release provenance | signed/SBOM/reproducible evidence where release tooling supports it; continue hardening |

## 8. Documentation completeness evaluation

Before this architecture-baseline branch, documentation was **not sufficient** for the current product:

- `docs/prd_trd_summary.md` described an early MLS2PLM-focused MVP and stated NumPy-first/default and multiple features as out-of-MVP that now coexist with broader protected-main scoring/rubric/model-selection functionality.
- no root `ARCHITECTURE.md` was present;
- no discoverable ADR directory/index was present;
- no integrated UML or logical ERD set was found;
- substantial architecture knowledge existed only in `AGENTS.md`, `CLAUDE.md`, specialized docs, PR bodies, handoff files, and project conversations.

This branch repairs that documentation topology by establishing authoritative PRD, TRD, architecture, ADR, UML, ERD, and traceability artifacts. The remaining work is to keep them synchronized as open PRs such as #566 integrate or are superseded.

## 9. Update rule

When a capability moves between `planned`, `open_pr`, and `implemented_on_main`, update this matrix in the integrating PR if the change materially affects architecture, ownership, a public contract, or a research-backed scientific claim.
