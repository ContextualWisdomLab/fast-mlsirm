# ADR-0001: Domain-neutral measurement boundary

Status: **Accepted**  
Date: 2026-08-09

## Context

`fast-mlsirm` has grown from an MLS2PLM-focused toolkit into a reusable measurement layer with assessment/rubric/scoring contracts, automated-scoring adapters, psychometric diagnostics, item/rater calibration and release evidence. ContextualWisdomLab also has downstream hosted/application bounded contexts, especially Psychometrics Commons.

Without an explicit boundary, the library can accidentally absorb participant/session state, identity, hosted persistence, UI and deployment logic. That would make the numerical core harder to install independently, create circular repository dependencies, and duplicate ownership already assigned to other CWL services.

The reusable core's scientific identity remains the simple-structure MLSIRM/MLS2PLM family and adjacent measurement contracts. Jeon, Jin, Schweinberger, and Baugh (2021) define the latent-space item-response interaction map; Kang and Jeon (2025) give the multidimensional extension and the relativity of conditional dependence; Molenaar and Jeon (2026) give the regularized joint-maximum-likelihood estimation strategy used by the point-estimate path. Score interpretation, fairness, and intended use remain governed by AERA, APA, and NCME (2014). ISO/IEC/IEEE 42010:2022 governs how this boundary is recorded as an architecture decision, not the psychometric likelihood.

## Decision

`fast-mlsirm` owns reusable domain-neutral measurement contracts and scientific/numerical capabilities:

- Assessment/Rubric/Scoring contracts and observations;
- calibration, CTT/IRT/MIRT, MLSIRM/MLS2PLM, facets, testlets and related kernels;
- model diagnostics, linking, DIF/invariance/fairness, CAT/ATA and G-theory;
- factor retention/model comparison, bifactor scoreability, rotation and recovery;
- automated-scoring and LLM-judge measurement primitives;
- governed rubric/item-bank primitives and portable scientific/audit reports.

The hosted product boundary belongs downstream. In particular `ContextualWisdomLab/psychometrics-commons` owns hosted/public/admin APIs, participant/session/response/consent/result lifecycle, product databases/migrations, tenant/resource authorization, UI/reference client behavior, deployment composition and research-release orchestration.

The dependency direction is:

```text
hosted/downstream product -> fast-mlsirm
```

and never:

```text
fast-mlsirm -> hosted product
```

Other CWL services are explicit optional integrations, not hidden implementation dependencies.

## Invariants and evidence

- The package must install and execute without Psychometrics Commons source or runtime.
- No product ORM/database, HTTP route, session/consent or UI type may become a required `fast_mlsirm` dependency.
- Cross-repository composition uses versioned APIs/contracts/immutable artifacts.
- Hosted product state must not be recreated under a library-local assessment runtime service.
- `AGENTS.md` and `CLAUDE.md` carry the same boundary.

## Consequences

Benefits:

- independent adoption and testing;
- clear MSA ownership and security authority;
- lower coupling between scientific evolution and product deployment;
- reusable measurement contracts across essay, RAG, enterprise-issue and other domains.

Costs:

- downstream adapters are required for persistence/transport;
- some end-to-end features require cross-repository integration tests rather than one monolith.

## Alternatives considered

1. **Make fast-mlsirm the hosted platform.** Rejected because it couples scientific kernels to application infrastructure.
2. **Keep only numerical functions and move all contracts downstream.** Rejected because versioned measurement/scoring contracts are reusable scientific primitives and must stay adjacent to the numerical interpretation they govern.

## Reversal conditions

Supersede this ADR only if the organization intentionally redefines repository bounded contexts and provides a migration plan preserving independent scientific/numerical reuse.

## Research and standards basis

The architecture-description concern (stakeholders, viewpoints, correspondence, and decision records) follows ISO/IEC/IEEE 42010:2022. The measurement methods this boundary exists to keep independently reusable are the simple-structure MLSIRM/MLS2PLM specialization documented in `docs/papers/mls2plm-canonical-equations.md`, not a silent claim of the full discrimination-vector MLS2PLM model.

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5

Molenaar, D., & Jeon, M. (2026). Regularized joint maximum likelihood estimation of latent space item response models. *Psychometrika, 91*, 335–359. https://doi.org/10.1017/psy.2025.10068

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.
