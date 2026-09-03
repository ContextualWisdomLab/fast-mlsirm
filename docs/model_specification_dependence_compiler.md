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
- `GeneralizedMixedStructure`: fixed/random-effect structure plus a typed
  `MembershipStructure`;
- `EstimationPlan`: estimator identity, backend ownership, implementation state,
  and the exact compiled candidate to which that evidence applies;
- `IdentificationContract`: required identification rules, verification state,
  and exact candidate scope;
- `RecoveryContract`: required known-truth recovery evidence and exact candidate
  scope.

`compile_dependence_candidates()` then attaches a separate
`DependenceStructure`. It never changes the base parameter blocks and never
falls back to the local-independent model when a dependence-aware candidate is
unsupported.

## Full candidate identity

`CandidateIdentity` is the single owner of structural model identity. It covers
response-family/formulation/scale/base parameter blocks, dimensional formulation
and dimension count, generalized-mixed formulation/fixed effects/random effects,
typed membership topology and weight authority, and dependence
kind/formulation/parameter blocks. Evidence and maturity flags are intentionally
excluded so a scientific specification keeps one identity while implementation
and validation advance.

Canonical JSON uses deterministic key ordering and separators. The stable ID is
therefore formulation-readable while remaining collision-resistant across the
full structural contract:

```text
<base_formulation>__<dependence_formulation>__spec_sha256_<full_identity_digest>
```

Changing dimensions, fixed/random effects, classification topology, membership
multiplicity, weight authority, classification axes, or dependence structure
changes the candidate identity. Capability evidence is keyed by this same exact
ID; evidence attached to one structural specification cannot promote another.

## Published candidate manifest

Every compiled candidate emits a versioned, self-digesting JSON-shaped
manifest. The current contract is:

```text
manifest_schema_id      fast_mlsirm.model_specification.candidate_manifest
manifest_schema_version 1.0.0
manifest_sha256          SHA-256(canonical manifest payload excluding this field)
```

The digest binds model identity, capability status, equation/citation evidence,
estimator/identification/recovery scope, typed membership semantics, and the
foreign temporal boundary marker. Consumers such as TEPP must bind to an
explicit released package/version plus this manifest contract; an open PR head
is not a published production dependency.

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

The dependence formulation portions of the IDs remain explicit:

```text
lsirm_jeon_et_al_2021_extension
mlsirm_kang_jeon_2025_extension
dlsjm_jin_jeon_2019_extension
```

The word `extension` is intentional. The cited papers ground the named
latent-space dependence families; they do not establish every possible
response-family × generalized-mixed × dependence coupling. Novel couplings
remain `research_candidate` until their own exact generative equation,
identification, Rust estimator, and true-parameter recovery evidence exist.

## Capability promotion gate

Every requested LSIRM/MLSIRM/DLSJM variant is materialized as exactly one of:

- `supported`: explicit generative-equation identity, exact-candidate-scoped Rust
  estimator, exact-candidate-scoped identification evidence, primary citation
  evidence that includes the canonical baseline formulation citation declared
  by the compiled dependence family, and exact-candidate-scoped passing recovery
  evidence are all present; when membership weights are model-estimated, their
  declared recovery metric is also present in that recovery contract;
- `research_candidate`: the coupling is representable but one or more support
  gates are missing;
- `unsupported`: the base kernel declares the dependence incoherent, or a
  structural prerequisite is impossible (for example MLSIRM on a
  unidimensional main-effect specification).

A candidate cannot become `supported` merely because its name is in the
registry. A base-kernel estimator, base identification proof, base recovery
study, or unrelated nonblank citation does not transfer to an
LSIRM/MLSIRM/DLSJM extension. `EstimationPlan`, `IdentificationContract`, and
`RecoveryContract` must each name the exact full candidate through
`applies_to_candidate_id`; documentary equation/citation evidence is looked up
through `evidence_by_candidate_id`, and the citation tuple must contain the
compiled `DependenceStructure.baseline_citations`. This prevents evidence for
one dimensional, mixed-membership, dependence, or unrelated research record
from promoting another represented model.

`evidence_by_candidate_id` is an admission boundary, not an extensible callback
surface. The public compiler accepts an exact built-in `dict` whose keys are
exact nonblank candidate IDs and whose values are exact `CapabilityEvidence`
objects, then copies that dictionary before constructing any candidate identity.
A custom mapping therefore cannot run a `get()` callback between identity
derivation and manifest assembly and mutate the structural objects that the
manifest repeats. This preserves one coherent structural snapshot per compiled
candidate instead of merely digesting an internally inconsistent payload.

## Generalized mixed-model boundary

`GeneralizedMixedStructure` is declarative and does not use a free-form
membership label. `MembershipStructure` separates three different questions:

- `MembershipClassification`: `hierarchical` versus `cross_classified`;
- `MembershipMultiplicity`: `single` versus `multiple` membership;
- `MembershipWeightAuthority`: `not_applicable`, `explicit_normalized`, or
  `model_estimated`.

Cross-classification and multiple membership are orthogonal. A
cross-classified structure names at least two classification axes; it may still
be single membership within each axis. Multiple membership must state who owns
the weights. Explicit normalized weights are observed/model-input quantities;
model-estimated weights require a named recovery metric before the candidate can
be promoted to `supported`. The compiler does not invent or normalize weights
and does not force an observation into a single parent.

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
  longitudinal leakage controls, time-varying memberships, and dynamic
  evolution. Compiled candidates carry only the boundary marker `tepp_owned`;
  no TEPP temporal ontology or transition equation is duplicated here.
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
multiple membership, or covariates. Model-estimated membership weights require
the formulation's own named weight-recovery metric; a generic RMSE on another
parameter block does not satisfy that contract.

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