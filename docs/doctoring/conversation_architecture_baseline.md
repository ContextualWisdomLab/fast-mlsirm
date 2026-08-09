# Conversation-to-architecture baseline doctoring

## Purpose

This record explains how the 2026-08 project research/design conversation was consolidated into the normative PRD, TRD, root architecture, ADRs, diagrams, and traceability matrix. It is not a transcript and does not make unmerged work appear released. The repository's protected code and accepted ADRs remain authoritative when a historical design note conflicts with live behavior.

## Reconciliation findings

Before this baseline, the repository already contained extensive feature-specific papers, specifications, plans, doctoring, `AGENTS.md`, `CLAUDE.md`, rubric-generation documentation, automated-scoring contracts, enterprise-issue adapters, recovery studies, and a managed changelog. However:

- `docs/prd_trd_summary.md` still described an early Python-first MLS2PLM MVP and listed ordinal/adaptive/rubric-related work as out of scope;
- no root `ARCHITECTURE.md` existed;
- no canonical ADR index/template existed;
- material architectural decisions were scattered across doctoring/plans/agent instructions;
- no canonical UML/C4-style diagram set or logical artifact ERD existed; and
- no authoritative conversation/work-family -> PRD/TRD/ADR/code/test traceability matrix existed.

The resulting baseline therefore treats the previous documentation as **substantive but fragmented and partially stale**, not as absent.

## Accepted research principles carried into architecture

### LLM judges are measurement instruments, not truth

Reference-free RAG evaluation, automated essay scoring, and enterprise issue evaluation share one architectural principle: an LLM/rater output is an observation with possible severity, discrimination, range, bias, drift, and disagreement. The system should preserve evidence and calibrate rater behavior rather than average raw verdicts and call the result ground truth.

Reference-free is not truth-free. Context-grounded faithfulness, world correctness, completeness, relevance, robustness, abstention, and citation support can require different evidence regimes and should not be collapsed into one label without a measurement argument.

### Multidimensional, bifactor, higher-order, testlet, many-facet, and latent-space structures are not substitutes

Correlated multidimensional models represent multiple substantive traits. Bifactor models separate a general factor from residual specific factors. Higher-order models place the general factor above first-order traits. Testlet/two-tier models address local dependence/method structure. Many-facet models separate rater/task/occasion effects. Latent-space terms model remaining local person/item or system/query interaction after the substantive structure is represented.

The architecture therefore requires relation-aware comparison, residual-dependence evidence, held-out prediction, recovery, and separate scoreability evidence instead of adopting the most flexible model by default.

### Parameter recovery is stronger evidence than correlation

A perfect positive correlation can coexist with additive or multiplicative scale bias. For estimator/scorer claims, the release evidence therefore prioritizes identified/aligned bias, MAE/RMSE, uncertainty coverage, probability/ICC/information recovery, decision calibration, and subgroup/DIF behavior. Correlation and rank agreement remain useful secondary evidence for association.

### Dynamic rubrics are an item-bank construction problem

The durable product opportunity is not a function that asks an LLM to write a one-off Likert rubric. The accepted direction is:

```text
Rubric / construct specification
 -> evidence-grounded blueprint
 -> bounded generation contract
 -> hostile provider validation
 -> semantic/content screening
 -> human/artificial-crowd pilot
 -> psychometric calibration
 -> governed item-bank revision
 -> linking/drift/retirement
```

Benchmark criteria are candidate-blind by default. Candidate-aware discovery can be useful diagnostically or for training, but must use cross-fitting or otherwise separate criterion discovery from final scoring.

### Measurement and decision utility are separate layers

Enterprise issue measurement can estimate severity, evidence, stakeholder disagreement, and rater effects. A business action priority such as expected net intervention value additionally requires outcome, intervention, cost, and organizational utility assumptions. That causal/decision policy is not silently embedded in a psychometric score and is not claimed as a core `fast-mlsirm` estimator.

### Rotation and factor retention require empirical selection

No one factor-retention method or rotation criterion is universally optimal. Candidate count/structure selection uses multiple evidence sources; rotation uses a criterion registry, deterministic multi-start, stable equivalence alignment, and criterion-neutral stability/recovery/theory evidence. Finite multi-start returns the best observed solution, not a proof of global optimality.

### Hierarchy and time are part of the data-generating design

When data are nested, cross-classified, multiple-membership, repeated, or longitudinal, that structure should be represented explicitly. A discrete occasion-step AR coefficient is not a continuous-time model merely because timestamps are present. Flattening contextual structure by default is an unacceptable atomistic simplification for models that claim contextual inference.

## Software architecture implications

1. Rust remains the numerical source of truth, with Python as public orchestration/validation/reporting and a reviewed PyO3 bridge.
2. `RubricSpecification`, `AssessmentSpec`, and scoring/evidence contracts are versioned, immutable, content-addressed artifacts shared by domain adapters.
3. Generated/provider data is untrusted until structural/provenance/source checks; semantic and psychometric validity remain later gates.
4. A logical item/measurement-bank entity model is documented without introducing a hosted database in the reusable core.
5. Heavy scientific studies are scheduled/manual/release evidence; bounded PR tests prevent formula/resource regressions without exhausting the development queue.
6. Psychometrics Commons remains the hosted product owner; `fast-mlsirm` is reusable and independently installable.
7. Security/privacy design prefers purpose-limited evidence, exact provenance, least privilege, bounded retention interfaces, and authorization over blanket PII masking that would destroy measurement utility.

## Standards and primary-source hierarchy

For scientific or governance claims, use this evidence order where feasible:

1. current official standards/specifications;
2. primary peer-reviewed methodological research;
3. official implementation/library documentation for interoperability contracts;
4. newer preprints for emerging methods, explicitly marked as preprints; and
5. secondary summaries only as navigation, not as the final scientific oracle.

Legacy packages such as `kaefa`, `aFIPC`, or `nonnest2` are not runtime or CI oracles. Published methods are reimplemented and validated independently.

## APA 7 reference set

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

Bray, T. (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259). Internet Engineering Task Force. https://doi.org/10.17487/RFC8259

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*, 581–612.

Kane, M. T. (2013). Validating the interpretations and uses of test scores. *Journal of Educational Measurement, 50*(1), 1–73. https://doi.org/10.1111/jedm.12000

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*, 799–826.

National Institute of Standards and Technology. (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (NIST AI 100-1). U.S. Department of Commerce. https://doi.org/10.6028/NIST.AI.100-1

National Institute of Standards and Technology. (2024). *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile* (NIST AI 600-1). U.S. Department of Commerce. https://doi.org/10.6028/NIST.AI.600-1

Preacher, K. J., Zhang, G., Kim, C., & Mels, G. (2013). Choosing the optimal number of factors in exploratory factor analysis: A model selection perspective. *Multivariate Behavioral Research, 48*(1), 28–56.

Rijmen, F. (2010). Formal relations and an empirical comparison among the bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal of Educational Measurement, 47*, 361–372.

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*(2), 137–150.

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*, 664–684.

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496.

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x

World Wide Web Consortium. (2024, December 12). *Web Content Accessibility Guidelines (WCAG) 2.2* (W3C Recommendation). https://www.w3.org/TR/WCAG22/

JSON Schema. (2022). *JSON Schema core: A media type for describing JSON documents* (Draft 2020-12). https://json-schema.org/draft/2020-12/json-schema-core

## Conservative interpretation

This documentation baseline makes the architecture explicit; it does not certify the software, complete every planned work family, or convert Draft PRs into released capabilities. The traceability matrix is intentionally allowed to say `Active`, `Partial`, or `Planned`. Closing those rows requires code and exact-head scientific/quality evidence, not another documentation assertion.
