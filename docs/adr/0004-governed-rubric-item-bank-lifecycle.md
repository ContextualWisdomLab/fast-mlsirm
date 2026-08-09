# ADR-0004: Governed rubric and item-bank lifecycle

Status: **Proposed**  
Date: 2026-08-09

## Context

The numerical core can analyze calibrated observations, but a defensible measurement system also needs to construct, screen, calibrate, assemble, version and retire the criteria/items being measured. Ad hoc `prompt -> LLM item -> use immediately` workflows mix item generation with validation and can introduce candidate leakage, redundant criteria, unverifiable evidence, drift and version ambiguity.

Protected `main` already contains canonical rubric-centered blueprint and provider-neutral generation-contract primitives. The complete governed item-bank lifecycle is not yet fully protected-integrated, so this ADR remains Proposed.

## Decision

Build the reusable lifecycle as:

```text
RubricSpecification
 -> Measurement Blueprint
 -> Generation Contract
 -> untrusted Generated Candidate
 -> Structural Validation
 -> Evidence/Semantic Screening
 -> Artificial-Crowd / Calibration Pilot
 -> Rust Psychometric Calibration
 -> Information/Content-Constrained Assembly
 -> Approved Item Bank
 -> Monitoring / DIF / Drift / Exposure
 -> Quarantine / Retirement / New Rubric Revision
```

### Generation modes

- **Benchmark mode:** candidate-blind; criteria are generated from task contract and independent evidence, not target-system answers.
- **Diagnostic mode:** candidate-aware discovery may be used only with cross-fitting or an equivalent separation between criterion discovery and scored candidates.
- **Training mode:** may evolve criteria separately but must not contaminate a fixed benchmark bank.

### Criterion design

The canonical model supports rich internal criterion contracts; adapters may compile them to external rubric formats. Atomic binary/nominal criteria are preferred where one independently verifiable decision exists. Holistic ordinal levels remain valid when the construct genuinely requires ordinal synthesis.

### Lifecycle states

A representative lifecycle is:

`draft -> audited -> screened -> pilot -> calibrated -> approved -> active -> suspended/quarantined -> retired`.

A production/approved revision is immutable. Semantic changes create a new rubric/item revision and require linking/recovery evidence when scores must remain comparable.

## Required screening dimensions

- construct alignment;
- criterion atomicity;
- answerability/applicability;
- evidence grounding and provenance;
- ambiguity;
- direction/polarity validity;
- distractor/option integrity where applicable;
- redundancy/local dependence;
- candidate leakage;
- language/domain bias and future DIF risk;
- execution cost/resource bounds.

## Calibration/assembly principles

Raw LLM-proposed weights are not psychometric item information. Once pilot data exist, calibrated item/factor information, fit, DIF, rater facets, residual dependence, content constraints, anchor/linking needs, cost and exposure guide assembly.

Safety/policy-critical criteria may be conjunctive gates rather than compensable score weights.

## Consequences

Benefits:

- closes the upstream gap between rubric design and psychometric calibration;
- makes generated items auditable and versionable;
- supports living item banks without silently changing score meaning;
- creates a differentiating closed loop for AI/assessment evaluation.

Costs:

- requires orchestration, screening and lifecycle APIs beyond current blueprint compilation;
- real-model pilots can be expensive and must use bounded provider orchestration;
- linking/monitoring adds stateful host requirements, though reusable artifact contracts remain library-owned.

## Alternatives considered

1. **Generate a 1–5 rubric per question and average scores.** Rejected as insufficiently decomposed and uncalibrated.
2. **Keep a permanently static item bank.** Rejected as the only strategy; fixed banks remain supported but cannot address evolving AI/evaluation domains.
3. **Generate criteria from the candidate being scored.** Rejected for benchmark use because it double-dips; permitted only in isolated/cross-fitted diagnostic/training modes.

## Acceptance before status becomes Accepted

- trusted parser/screening path protected-integrated;
- at least one end-to-end candidate -> pilot -> Rust calibration workflow;
- governed item-bank revision/lifecycle contract;
- DIF/drift/linking evidence for version changes;
- realistic offline and bounded live-model tests;
- documentation and release evidence.

## References

Hashemi, H., Eisner, J., Rosset, C., Van Durme, B., & Kedzie, C. (2024). *Initial nugget evaluation results for the TREC 2024 RAG Track with the AutoNuggetizer framework*. arXiv:2411.09607.

Hashemi, H., et al. (2024). LLM-Rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. *Proceedings of ACL 2024*.

Shankar, S., Zamfirescu-Pereira, J. D., Hartmann, B., Parameswaran, A. G., & Arawjo, I. (2024). Who validates the validators? Aligning LLM-assisted evaluation of LLM outputs with human preferences. *Proceedings of UIST 2024*.
