# Model specification and dependence candidate compiler

Status: bounded Model Specification domain contract; candidate compilation only.

`fast_mlsirm.model_specification` separates a conditional response kernel from
its dimensional, generalized mixed-model, and residual-dependence structure.
The compiler is deliberately non-numerical: it records model intent and
scientific maturity before any Rust estimator is selected or implemented.

## Domain boundary

The **Core subdomain** remains reusable psychometric measurement and numerical
inference. Model Specification is a supporting domain responsibility that
creates immutable candidate contracts for that core. It does not evaluate a
likelihood or infer parameters.

A base `ModelSpecification` composes these Value Objects:

- `ResponseKernel`: response-family/formulation identity plus base parameter
  blocks and declarative dependence compatibility;
- `DimensionalStructure`: main-effect dimensional formulation;
- `GeneralizedMixedStructure`: fixed/random-effect and membership structure;
- `EstimationPlan`: estimator identity, backend ownership, implementation state,
  and the exact formulation to which that evidence applies;
- `IdentificationContract`: required identification rules, verification state,
  and exact formulation scope;
- `RecoveryContract`: required known-truth recovery evidence and exact
  formulation scope.

`compile_dependence_candidates()` then attaches a separate
`DependenceStructure`. It never changes the base parameter blocks and never
falls back to the local-independent model when a dependence-aware candidate is
unsupported.

## Dependence Ubiquitous Language

The compiler keeps three residual-dependence concepts distinct.

`LSIRM` denotes residual person-item interaction represented by person and item
positions in one interaction space, with response tendency related to their
latent distance. Jeon et al. (2021) is the baseline formulation reference.

`MLSIRM` denotes the multidimensional-main-effect latent-space extension. Main
factor/loadings and residual interaction geometry remain separate. A
unidimensional base therefore still produces an MLSIRM candidate record, but
that record is `unsupported` with
`multidimensional_main_effects_required` rather than disappearing from the
capability graph. Kang and Jeon (2025) is the baseline formulation reference.

`DLSJM` denotes the doubly latent-space joint-model perspective of Jin and Jeon
(2019): local item dependence and local person dependence are separate
relational structures with separate item- and person-space parameter blocks.
DLSJM is not an alias for LSIRM, MLSIRM, multilevel IRT, or a generic network
random effect.

The stable candidate identifiers are formulation-qualified:

```text
<base_formulation>__lsirm_jeon_et_al_2021_extension
<base_formulation>__mlsirm_kang_jeon_2025_extension
<base_formulation>__dlsjm_jin_jeon_2019_extension
```

The word `extension` is intentional. The cited papers ground the named
latent-space dependence families; they do not establish every possible
response-family × generalized-mixed × dependence coupling. Novel couplings
remain `research_candidate` until their own exact generative equation,
identification, Rust estimator, and true-parameter recovery evidence exist.

## Capability promotion gate

Every requested LSIRM/MLSIRM/DLSJM variant is materialized as exactly one of:

- `supported`: explicit generative-equation identity, formulation-scoped Rust
  estimator, formulation-scoped identification evidence, primary citation, and
  formulation-scoped passing recovery evidence are all present;
- `research_candidate`: the coupling is representable but one or more support
  gates are missing;
- `unsupported`: the base kernel declares the dependence incoherent, or a
  structural prerequisite is impossible (for example MLSIRM on a
  unidimensional main-effect specification).

A candidate cannot become `supported` merely because its name is in the
registry. A base-kernel estimator, base identification proof, or base recovery
study does not transfer to an LSIRM/MLSIRM/DLSJM extension: each of those three
contracts must name the exact compiled candidate ID through
`applies_to_formulation_id`. This prevents evidence for one extension from
promoting another extension with a different likelihood or dependence geometry.

## Generalized mixed-model boundary

`GeneralizedMixedStructure` is declarative. It preserves fixed effects, random
effects, and membership semantics as an orthogonal axis so future compilers can
compose nested, crossed, cross-classified, multiple-membership, and other
scientifically identified structures without copying the response kernel or the
dependence implementation into family-specific branches.

This first slice does not implement generalized-mixed likelihood arithmetic.
All future production vector, matrix, likelihood, gradient, integration,
uncertainty, simulation, and recovery arithmetic remains Rust-owned.

## Context map

- `fast-mlsirm` owns this reusable model-specification Published Language and
  all future reusable psychometric numerical kernels.
- `contextual-orchestrator` owns any LLM provider execution, routing, fallback,
  credentials, judge generation, verifier/adjudicator orchestration, and model
  call provenance. This package consumes resulting observations/provenance only.
- `LineageWeave` is a downstream/foreign bounded context for construct/rubric
  meaning, source-evidence binding, pilot lifecycle, administrator workflows,
  and buyer-facing interpretation. It may translate through a versioned ACL;
  fast-mlsirm does not import its product types.
- `TEPP` owns temporal-event semantics, changing temporal state, event graphs,
  longitudinal leakage controls, and dynamic evolution. Compiled candidates
  carry only the boundary marker `tepp_owned`; no TEPP temporal ontology or
  transition equation is duplicated here.
- `psychometrics-commons` remains the downstream hosted product owner for
  persistence, participant/session lifecycle, authorization, and publication.

## Evidence and recovery direction

A later estimator PR that promotes one candidate to `supported` must attach
model-appropriate known-truth evidence. For LSIRM/MLSIRM this includes aligned
latent-position/distance recovery, interaction-strength recovery, map stability,
coverage where defined, latent-dimension sensitivity, residual-dependence
reduction, and comparison with substantive testlet/MIRT/facet/multilevel
alternatives. For DLSJM it additionally requires separate item-space and
person-space distance/position recovery, false-dependence rates under independent
data, alignment invariance, and checks that the geometry is not absorbing
omitted multidimensionality, common stimuli, rater effects, group structure,
multiple membership, or covariates.

Correlation by itself is not recovery evidence.

## Primary research basis

Jin, I. H., & Jeon, M. (2019). A doubly latent space joint model for local item
and person dependence in the analysis of item response data. *Psychometrika,
84*(1), 236–260. https://doi.org/10.1007/s11336-018-9630-0

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved
item–respondent interactions: A latent space item response model with interaction
map. *Psychometrika, 86*(2), 378–403.
https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models:
A note on the relativity of conditional dependence. *Psychometrika, 90*(2),
799–826. https://doi.org/10.1017/psy.2025.5

The repository cites and links these works; this change does not redistribute
publisher PDFs.
