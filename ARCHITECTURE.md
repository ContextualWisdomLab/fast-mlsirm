# fast-mlsirm Architecture

Status: **Authoritative architecture description**  
Repository: `ContextualWisdomLab/fast-mlsirm`  
Last reviewed: 2026-08-09

This document describes the current and intended architecture of `fast-mlsirm` using the concerns and viewpoints of ISO/IEC/IEEE 42010:2022. It is the root navigation point for product requirements, technical requirements, ADRs, UML/ERD diagrams, and research-to-code traceability.

## 1. Mission and boundary

`fast-mlsirm` is a reusable, domain-neutral psychometric measurement library. It owns scientific/numerical measurement truth and reusable contracts; it does not own a hosted assessment application's runtime state.

```text
Downstream products and services
  Psychometrics Commons / independent Python callers / research pipelines
                              |
                     versioned public contracts
                              v
+-------------------------------------------------------------------+
|                         fast-mlsirm                               |
|                                                                   |
| Assessment/Rubric/Scoring contracts  ->  validation/orchestration |
|              |                                      |             |
|              v                                      v             |
| item/rater/response evidence -> psychometric model selection      |
|              |                                      |             |
|              +---------------> Rust numerical core <+             |
|                                      |                            |
|                            PyO3 / typed results                    |
|                                      |                            |
|               reports / recovery / release evidence               |
+-------------------------------------------------------------------+
                              |
          optional explicit integrations, never hidden coupling
                              v
 contextual-orchestrator / TEPP / Gyeot / semantic-data-portal / ...
```

### Owned bounded context

- assessment/rubric/scoring contracts;
- item/rater/response observations and calibration handoff;
- CTT/IRT/MIRT and MLSIRM-family numerical functions;
- testlet, many-facet and model-diagnostic primitives;
- factor/model selection, bifactor scoreability, rotation and recovery;
- DIF/invariance/fairness, linking/equating, G-theory, CAT/ATA;
- automated-scoring and LLM-judge validation primitives;
- governed rubric/item-bank primitives;
- deterministic scientific/audit/report/release evidence.

### Explicitly outside the bounded context

- product HTTP/admin APIs;
- participant/session/consent/result persistence;
- identity/federation credentials;
- hosted tenant/database migrations;
- end-user UI/application navigation;
- model-provider secret stores;
- product deployment control plane.

`ContextualWisdomLab/psychometrics-commons` is the canonical hosted assessment product and a downstream consumer. The dependency direction is downstream -> `fast-mlsirm`; never the reverse.

## 2. Architecture drivers

1. **Scientific defensibility:** model interpretation must be tied to identification, fit, recovery, invariance and uncertainty evidence.
2. **Reproducibility:** content-addressed contracts and immutable revisions must make analyses independently reconstructible.
3. **Performance:** production psychometric arithmetic is Rust-first with low-context-switch CPU parallelism and parity-verified GPU paths where material.
4. **Safety:** untrusted provider/source/user data is bounded and fail-closed; model/review/security uncertainty is not silently converted into success.
5. **Composability:** the package remains independently installable while exposing stable versioned contracts to CWL services and third parties.
6. **Explainability/auditability:** exact numerical values, provenance, model relation, convergence and interpretation boundaries remain machine-readable.
7. **Evolution:** rubrics, items, models and calibration artifacts evolve by version/supersession rather than in-place semantic mutation.

## 3. Architectural views

### 3.1 Package/component view

See [`docs/uml/component.puml`](docs/uml/component.puml).

```text
Python API / CLI
      |
      +-- assessment & rubric contracts
      +-- scoring / essay / enterprise adapters
      +-- validation / model comparison / reporting
      +-- orchestration + bounded input validation
      |
      v
PyO3 binding registry
      |
      v
Rust mlsirm-core
  likelihood + gradients
  fitting / fit statistics
  facets / agreement
  linking / CAT / ATA
  rotation / scoreability / future model kernels
      |
      +-- CPU parallel execution
      +-- GPU kernels where supported
```

The Python layer may keep transparent reference/fallback implementations only when they are explicitly governed by parity contracts. It may not become a second silently different production psychometric engine.

### 3.2 Contract/data-flow view

See [`docs/uml/scoring-sequence.puml`](docs/uml/scoring-sequence.puml).

A governed scoring path is:

```text
AssessmentSpec + RubricSpecification
                 |
                 v
             ScoringRequest
                 |
                 v
       Human / AI / External Engine
                 |
                 v
            ScoreObservation
                 |
                 v
      criterion/rater task handoff
                 |
                 v
          Rust calibration
                 |
                 v
 validation / fairness / adjudication / report
```

Every trust boundary rechecks content identity rather than trusting display handles or cached parent objects.

### 3.3 Rubric/item-bank view

```text
approved rubric
   -> deterministic blueprint
   -> provider-neutral generation contract
   -> untrusted candidate
   -> structural/evidence/semantic screening
   -> calibration-system/artificial-crowd pilot
   -> Rust psychometric calibration
   -> item-bank approval
   -> active monitoring (fit/DIF/drift/exposure)
   -> quarantine/retirement or new rubric revision
```

Candidate-blind generation is the default for benchmark/evaluation banks. Candidate-aware discovery is a separate proposed mode and requires cross-fitting or equivalent anti-leakage design.

### 3.4 Model-selection view

See [`docs/uml/model-selection-sequence.puml`](docs/uml/model-selection-sequence.puml).

Model selection is intentionally multi-stage:

1. determine substantive factor-retention candidates;
2. classify structural relation from actual constraints;
3. use relation-appropriate inferential comparison;
4. compare held-out cluster-aware predictive evidence;
5. inspect residual dependence and DIF/invariance;
6. inspect scoreability and factor/rotation stability;
7. confirm true-structure/parameter recovery under realistic simulation;
8. choose the simplest model meeting interpretation requirements.

Bifactor, higher-order, testlet, two-tier, many-facet and latent-space structures are not interchangeable names. Each answers a different scientific question.

### 3.5 Lifecycle view

See [`docs/uml/item-bank-state.puml`](docs/uml/item-bank-state.puml).

Reusable governed artifacts use explicit immutable/superseding lifecycles. A representative item lifecycle is:

```text
draft -> audited -> screened -> pilot -> calibrated -> approved -> active
                                             |              |
                                             v              v
                                         quarantined     suspended
                                             |              |
                                             +-------> retired
```

Published/approved revisions do not silently mutate. A changed rubric/task/calibration produces new semantic identity and linking/recovery evidence as required.

### 3.6 Deployment/composition view

See [`docs/uml/deployment.puml`](docs/uml/deployment.puml).

`fast-mlsirm` is delivered as a Python package with a compiled Rust extension. It may be embedded in a CLI, notebook, batch worker, service, or hosted product. Those hosts own transport, authentication, tenancy, persistence and deployment.

Optional CWL integrations are explicit:

- Psychometrics Commons: hosted assessment lifecycle and APIs.
- Keyverse: identity/federation.
- Gyeot: EMA/ESM collection.
- TEPP: temporal/event/relationship analytics.
- contextual-orchestrator: bounded provider-neutral LLM orchestration.
- pg-llm-batch: bulk asynchronous model work when selected by a host.
- semantic-data-portal: research catalog/release provenance.
- EgressWeave: controlled external egress in hosts that use it.

No service may access another service's application database through `fast-mlsirm`.

## 4. Logical domain model

See [`docs/erd/domain-model.puml`](docs/erd/domain-model.puml).

The ERD is a **logical reusable-domain model**, not a prescription that `fast-mlsirm` owns a hosted relational database. It documents identity and cardinality among reusable records such as:

- Assessment specification/version;
- Rubric specification/level;
- Item blueprint/candidate/revision;
- Scoring request/observation/result;
- Engine/rater descriptor;
- calibration design/report;
- item-bank entry/calibration history;
- model-comparison/recovery evidence.

Hosts may persist these artifacts in different stores as long as the public serialization and provenance contracts remain valid.

## 5. Numerical architecture

### 5.1 Rust source of truth

The Rust core owns production numerical algorithms. Python performs:

- validation and bounded materialization;
- provider/domain orchestration;
- NumPy marshaling;
- transparent reference calculations where explicitly retained;
- report construction.

### 5.2 Backend model

The public backend axis remains `{auto, rust, numpy}` for currently supported APIs, with Rust the preferred resolved backend when the compiled extension is available. GPU is a device choice under the Rust path, not a third formula implementation.

### 5.3 Identification-aware parity

Parity is checked at the mathematical invariant that is actually identified:

- raw scalar/vector equality where identified;
- Procrustes-aligned loadings/coordinates when rotation is arbitrary;
- pairwise distance matrices for latent-space geometry when coordinates are rotation/reflection/translation invariant;
- linked/scaled parameter errors only after scale alignment.

## 6. Scientific architecture

### 6.1 Reference-free evaluation

Reference-free does not mean truth-free. The evidence regime determines what may be claimed. Context-supported faithfulness may be measured without an external truth reference; world correctness and absolute corpus recall require stronger evidence.

LLM judges are fallible raters. Model family, version, prompt, order/occasion and assignment design may create severity, bias, discrimination or drift and shall be retained when interpretation depends on them.

### 6.2 Multidimensional and bifactor structure

Multidimensional traits model substantive quality dimensions. Bifactor models separate a candidate general factor from specific factors. Testlets/local-dependence effects model shared stimulus/question dependence. Latent space is a residual interaction mechanism to be added only after substantive/facet/testlet structure is modeled and predictive/recovery evidence supports it.

### 6.3 Model relation

The architecture does not allow a normal-theory preference merely because two log-likelihoods differ. Relation classification and formal distinguishability are decision gates.

### 6.4 Recovery

True-parameter recovery is a core scientific release mechanism. Correlation may be reported as supplementary order-preservation evidence; bias, RMSE, coverage, convergence and function/information recovery provide direct accuracy evidence.

## 7. Security, privacy and trust boundaries

### 7.1 Input/provider boundaries

- bound before allocate/read/materialize;
- closed schema for provider-generated structured content;
- reject duplicate keys and non-finite JSON numbers;
- verify source/evidence spans against exact source revisions;
- sanitize/redact untrusted exception text;
- do not embed raw secrets or uncontrolled source text into identifiers/evidence logs.

### 7.2 Repository/release boundaries

- least-privilege GitHub Actions;
- immutable action pins where practical;
- central security/SAST/dependency gates;
- no self-modifying write-capable PR workflow;
- exact-head evidence and stale-head refusal;
- release only from protected integrated evidence.

### 7.3 PII/data ownership

The core library should not require blanket masking that destroys measurement semantics. Prefer purpose-limited data contracts, opaque identities, minimal fields, hashed/content-addressed provenance, host-owned encryption/access control, and separation of identity-bearing hosted data from reusable measurement artifacts.

## 8. Quality attributes

The principal ISO/IEC 25010:2023-aligned concerns are:

- functional suitability: explicit scientific/API contracts;
- performance efficiency: bounded resources, Rust/CPU/GPU design;
- compatibility: independent package plus versioned integrations;
- interaction capability/accessibility: understandable APIs, accessible reports;
- reliability: deterministic provenance, timeouts, recovery, fail-closed semantics;
- security: untrusted-input boundaries, least privilege, supply-chain evidence;
- maintainability/flexibility: modular kernels, canonical binding registry, ADRs and traceability;
- safety: no unsupported psychometric/causal/high-stakes claims.

## 9. Architecture decisions

The ADR index is [`docs/adr/README.md`](docs/adr/README.md). Major accepted/proposed decisions cover:

1. domain-neutral library versus hosted product ownership;
2. Rust-first numerical ownership and PyO3 boundary;
3. canonical content-addressed assessment/rubric/scoring contracts;
4. governed rubric/item-bank lifecycle and candidate leakage control;
5. humans/LLMs as calibratable raters for automated scoring;
6. relation-safe model-selection hierarchy;
7. multilevel/multiple-membership/temporal first-class structure;
8. true-parameter recovery as scientific CI evidence;
9. adaptive rotation without a universal-best criterion;
10. LLM orchestration/credential boundaries.

## 10. Architecture conformance

A material code change conforms when:

- it respects the bounded-context ownership rules;
- numerical ownership remains consistent with the Rust/Python contract;
- serialized semantic changes receive versions/provenance and migration/compatibility evidence;
- new model claims have identification/recovery evidence;
- trust boundaries remain fail closed;
- relevant PRD/TRD requirement IDs and ADRs are updated or explicitly unaffected;
- tests and release evidence cover the changed contract.

A lightweight machine-checkable documentation contract is maintained in `tests/test_architecture_documentation_contract.py`.

## 11. References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

International Organization for Standardization. (2023). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software Quality Requirements and Evaluation (SQuaRE)—Product quality model*.

International Organization for Standardization. (2023). *ISO/IEC 42001:2023 Information technology—Artificial intelligence—Management system*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.

World Wide Web Consortium. (2024). *Web Content Accessibility Guidelines (WCAG) 2.2*.
