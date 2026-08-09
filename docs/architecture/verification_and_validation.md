# Verification and validation architecture

## Status

This document is part of the proposed canonical architecture baseline. It distinguishes protected-main implementation from active pull requests and roadmap obligations. It does not declare an unmerged model or workflow operational.

## Purpose

`fast-mlsirm` requires two different evidence classes:

- **verification:** the software implements its declared formulas, contracts, bounds, and interfaces correctly; and
- **validation:** the resulting scores, parameters, classifications, reports, and uses support the intended interpretation under the target population, design, domain, and decision context.

A green unit test is not construct validation. A high human–AI correlation is not agreement or parameter recovery. A fitted bifactor model is not permission to report general or specific scores. A generated item that satisfies JSON Schema is not a valid assessment item.

## Evidence flow

```text
Requirement / scientific claim / threat
                    │
                    ▼
        Contract, equation, or control
                    │
                    ▼
      Deterministic verification evidence
                    │
                    ├── unit / property / metamorphic tests
                    ├── Rust↔reference / CPU↔GPU parity
                    ├── security / resource / concurrency tests
                    ├── schema / provenance / migration tests
                    └── package / reinstall / release evidence
                    │
                    ▼
       Scientific and operational validation
                    │
                    ├── true-parameter bias / MAE / RMSE
                    ├── uncertainty and interval coverage
                    ├── model-selection and recovery confusion matrix
                    ├── fit, local dependence, DIF, invariance and drift
                    ├── scoreability and decision consistency
                    ├── held-out population/task/rater/domain evidence
                    └── human audit and consequential-use evidence
                    │
                    ▼
     Supported interpretation and bounded release claim
```

## Verification layers

| Layer | Required evidence | Failure meaning |
|---|---|---|
| Canonical contracts | exact types, bounded collections, immutable canonical serialization, revision fingerprints, replay rejection | artifact identity or cross-service interpretation is unsafe |
| Numerical kernels | analytic/finite-difference checks, independent oracle, Rust/reference parity, CPU/GPU parity, numerical boundary tests | implementation does not faithfully compute the declared model |
| Estimation | objective monotonicity or declared stopping behavior, convergence status, multistart handling, singularity/overflow checks | returned estimate may be computationally invalid |
| Data design | connectedness, identifiability, missingness, cluster/testlet/rater coverage, revision linkage | target parameter is not estimable from the administered design |
| Security/privacy | hostile input, allocation ceilings, path/descriptor binding, credential isolation, source-text absence, provenance replay | evidence or runtime boundary can be subverted or leak data |
| Packaging/release | wheel/sdist build, clean reinstall, public import, SBOM, artifact digest, provenance, migration and rollback | source tests do not prove the offered artifact works |
| Documentation | PRD/TRD/ADR/UML/ERD/traceability consistency and maturity labels | users or buyers can rely on a false capability or boundary |

## Psychometric validation layers

### Parameter recovery

After scale, sign, permutation, rotation, Procrustes, or linking alignment appropriate to the model, simulation studies report at least:

- bias;
- mean absolute error;
- root mean squared error;
- standard-error bias where uncertainty is returned;
- confidence or credible interval coverage;
- convergence and admissible-solution rate;
- recovery by parameter class and substantive condition.

Correlation may be reported only as supplementary rank/association evidence.

### Structural recovery and model selection

When several structures are candidates, simulations cross the generating and fitted models and report selection confusion matrices. Relationship classification precedes testing:

- regular nested: regular or robust likelihood-ratio procedure;
- boundary or singular nested: boundary-aware or parametric-bootstrap procedure;
- strictly non-nested or overlapping: formal distinguishability before Vuong preference;
- unknown relationship: no automated winner.

Selection also considers held-out or cluster-aware predictive evidence, residual local dependence, factor/score recovery, DIF/invariance, scoreability, and interpretability.

### Scoreability

A model that fits may still produce unsupported scores. Relevant evidence includes:

- marginal reliability and conditional standard error;
- bifactor ECV, PUC, omega-hierarchical, omega-hierarchical-subscale, determinacy and construct replicability;
- higher-order proportionality and first-order score precision;
- testlet variance and residual-dependence reduction;
- facet connectedness, severity separation, consistency and range use;
- external and incremental validity of reported subscores.

No universal threshold is embedded without a declared policy and evidence basis.

### Generalization and fairness

Split units follow the intended claim. Random cell splits are prohibited when they leak respondent, query, task, testlet, rater, model family, or domain information across training and validation. Applicable designs include:

- leave-person/respondent/system-out;
- leave-item/task/query/testlet-out;
- leave-rater or rater-family-out;
- leave-domain/language/group/time-window-out;
- forward-chaining temporal validation.

DIF, measurement invariance, subgroup bias/RMSE/coverage, conditional error, calibration and decision consistency are reported separately from aggregate association.

## Generated rubric and item validation

The authoring lifecycle has distinct gates:

```text
RubricSpecification
  → Blueprint
  → GenerationContract
  → hostile provider output
  → structural/provenance validation
  → semantic/evidence/bias/leakage screening
  → artificial-crowd and/or human pilot
  → Rust calibration and model diagnostics
  → approval and immutable item-bank version
  → drift/DIF/exposure monitoring
  → suspension, revision or retirement
```

Structural conformance is necessary but does not establish answerability, construct alignment, source entailment, discrimination, fit, fairness, or operational utility.

## Automated scoring validation

Human, LLM, ML, rules, and external engines are represented as fallible raters or observations. Validation separates:

- exact and adjacent agreement;
- quadratic weighted kappa;
- absolute agreement or concordance where continuous scores are compared;
- severity, criterion interaction, consistency and range restriction;
- abstention and insufficient-evidence behavior;
- calibration, drift, DIF and subgroup error;
- human adjudication and override patterns;
- downstream decision error.

Raw human scores are not assumed to be error-free truth.

## Release evidence classes

A release candidate binds the following to one protected commit and artifact set:

1. deterministic CI and coverage;
2. Rust/PyO3/package and explicit backend execution;
3. security and supply-chain gates;
4. method-specific recovery and validation required by changed behavior;
5. documentation and traceability consistency;
6. wheel/sdist hashes, SBOM and provenance;
7. clean-environment reinstall/import and realistic examples;
8. migration/rollback and compatibility evidence;
9. independent review and release acceptance.

Evidence from a predecessor head, synthetic merge only, skipped job, status-only context, missing artifact, or different package build does not transfer.

## Maturity matrix

| Capability | Protected-main evidence | Required future evidence |
|---|---|---|
| MLS2PLM simple-structure simulation and point estimation | Implemented; Rust primary path and reference parity | broader model/population recovery and continued backend evidence |
| Rubric and assessment contracts | Implemented contract slices | semantic screening, artificial crowd, calibration and governed bank |
| Generated-item provider validation | Partial/active work | hostile-output parity, semantic and psychometric acceptance |
| Automated essay validation | Partial/active work | range-use, rater-characteristic and held-out validity evidence |
| Multilevel/multiple-membership/time contracts | Active PR/proposed estimator path | identified Rust likelihood, recovery, CPU/GPU parity and drift evidence |
| Factor retention and full structure selection | Planned/partial supporting diagnostics | unified relation-aware selection and true-structure recovery |
| Hosted tenant, consent, persistence and administration | Downstream | Psychometrics Commons product evidence |

## Traceability rule

Every substantive requirement is linked in `docs/requirements_traceability.md` to:

- governing ADR and scientific source;
- implementation path or active issue/PR;
- deterministic verification;
- scientific/operational validation;
- maturity state;
- release evidence when accepted.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

International Organization for Standardization & International Electrotechnical Commission. (2023). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software quality requirements and evaluation (SQuaRE)—Product quality model*.

Kane, M. T. (2013). Validating the interpretations and uses of test scores. *Journal of Educational Measurement, 50*(1), 1–73. https://doi.org/10.1111/jedm.12000

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
