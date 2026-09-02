# Temporal psychometric research and bounded-context ownership

Scientific status: Proposed / evolving implementation  
Ownership status: Accepted via ADR-0028  
Reviewed: 2026-09-02

## Why this record exists

The repository already contains legitimate multilevel, multiple-membership, repeated-occasion, discrete autoregressive, and continuous-time psychometric research. That scientific lineage does not imply that `fast-mlsirm` owns the temporal/event domain itself.

TEPP owns temporal/event composition and semantics: event ontology and graph construction, valid/system/event-time meaning, event ordering, changing-membership history, longitudinal split/leakage policy, and domain interpretation of change or drift. fast-mlsirm owns reusable time-indexed psychometric numerical kernels over explicit supplied person/occasion/time carriers when the model equation and identification contract are themselves part of the psychometric estimand.

The temporal-method papers below ground model structure, estimation, or recovery questions. They do not transfer temporal/event semantic authority from TEPP to fast-mlsirm. A numerical elapsed-time parameter is therefore not permission to recreate event ontology, temporal-validity rules, or changing-membership history locally.

## Research-to-implementation interpretation

The retained scientific program separates at least four claims:

1. static multilevel, cross-classified, and weighted multiple-membership psychometric structure;
2. repeated-occasion numerical structure whose ordering/time carriers are supplied explicitly;
3. discrete autoregressive or continuous-time state equations whose transition parameters are part of a specified psychometric model; and
4. TEPP-owned temporal/event semantics used to compose and validate the upstream event history from which those carriers may be produced.

The first Rust state layer described by ADR-0019 is an independent per-respondent OLS trend plus caller-supplied discrete AR predictor. The ADR-0020 slice is a separate joint MAP hierarchical continuous-time AR(1) Rasch numerical model over explicit person-occasion states and elapsed-time inputs. Neither implementation makes fast-mlsirm authoritative for event meaning, event graph construction, valid/system time, changing-membership history, or leakage policy.

Research support remains bounded to the exact estimand. Fox and Glas (2001) support multilevel IRT; Browne, Goldstein, and Rasbash (2001) support multiple-membership multiple-classification structure; Jeon and Rabe-Hesketh (2016) support an autoregressive longitudinal item-analysis model; Oravecz, Tuerlinckx, and Vandekerckhove (2011) support hierarchical continuous-time latent stochastic modeling. Those sources do not justify claiming that one repository owns all temporal semantics, nor do they establish every generalized-mixed × temporal × dependence combination.

Support promotion still requires an exact formulation, identification conditions, Rust implementation, true-parameter recovery appropriate to the estimand, deterministic seed evidence, bias/MAE/RMSE, interval/SE coverage where uncertainty is claimed, convergence, and CPU/GPU parity when a GPU path is promoted. Correlation is supplementary rather than a substitute for recovery.

## Context Fabric and EA projection

A TEPP-originated temporal design enters fast-mlsirm only through the versioned immutable Anti-Corruption Layer governed by ADR-0028. Direct TEPP database access, hidden TEPP runtime dependencies, and cross-service SQL is prohibited.

Architecture/package/backend/toolchain/consumer-lifecycle facts may be projected toward `ContextualWisdomLab/enterprise-architecture-core` only through an immutable released context-graph-contracts contract carrying versioned Context Assertion / CloudEvent / conformance identity and provenance. An unreleased sibling branch or PR head is not production authority.

Estimator values, latent scores, DIF/fit diagnostics, bias, RMSE, coverage, and scientific-validity evidence are not EA-authoritative facts. They remain measurement/scientific evidence in the owning psychometric context and must not be duplicated into the EA Decision Plane as architecture truth.

## Primary research basis

- Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*(2), 271–288. https://doi.org/10.1007/BF02294839
- Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124. https://doi.org/10.1177/1471082X0100100202
- Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for longitudinal item analysis. *Psychometrika, 81*(3), 830–850. https://doi.org/10.1007/s11336-015-9489-2
- Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876
- Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011). A hierarchical latent stochastic differential equation model for affective dynamics. *Psychological Methods, 16*(2), 468–490. https://doi.org/10.1037/a0024375
- Uto, M. (2023). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928. https://doi.org/10.3758/s13428-022-01997-z
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

## Traceability rule

`docs/traceability/research-basis.md` remains the broad research-to-architecture baseline. Its section on multilevel, multiple-membership, and temporal measurement describes the scientific/model program; this record is the code-current ownership interpretation for that section under accepted ADR-0028. Future scientific revisions may change equations or evidence requirements, but they do not move the TEPP/fast-mlsirm boundary without a new reviewed architecture decision and matching fitness tests.
