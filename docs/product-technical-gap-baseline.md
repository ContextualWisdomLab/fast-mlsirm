# Product and technical gap baseline

Status: **Non-authoritative point-in-time product-completion inventory**  
Protected-product basis: `main@45627700c26c29bca150896a9519a9b7426acb56`  
Observation date: 2026-09-02  
Predecessor baseline artifact used for this refresh: branch head `3a3865f40da12211898c97cbd47e7460381736ae`, blob `0f514caa4f4c5cabbb8522c1da79475d854e030b`

This document is a commercialization and technical-gap inventory, not runtime authority. Capability is authoritative only after the relevant source is integrated into the protected product branch and the required scientific, package, security, review, SBOM/provenance and release evidence is green on one unchanged exact head. Open PRs, successful predecessor checks and draft documentation are evidence inputs, not released product claims.

## 1. Product boundary

`fast-mlsirm` is the canonical reusable psychometric numerical engine for LSIRM/MLSIRM/MLS2PLM and adjacent dependence/IRT families. Production likelihood, optimization, scoring, information/uncertainty, covariance/correlation, simulation/recovery and other result-affecting vector/linear/matrix arithmetic belongs in Rust/PyO3. Python is limited to validation, provenance sealing, marshalling, orchestration, reporting and explicit reference/parity surfaces.

The internal architecture is organized around the bounded contexts **Model Specification**, **Estimation**, **Scoring**, **Diagnostics**, **Simulation-Recovery**, **Compute Backend** and **Public Binding**. Cross-context dependencies should use explicit contracts instead of implementation imports. Temporal/event semantics and composition remain TEPP-owned; this repository may own reusable time-indexed psychometric numerical kernels over explicit supplied time/occasion carriers, but it does not own TEPP event ontology, clocks or temporal workflow semantics.

Rasch and generic 1PL are not synonyms in product claims. New 2PL/3PL/4PL, bifactor, higher-order, two-tier, multifacet/multifactor, cross-loading, DIF, CAT/ATA and dependence-family support is promotable only when the exact formulation has primary-research grounding, identification constraints, deterministic public contract and true-parameter recovery evidence appropriate to the claimed use.

## 2. Commercial merge and release gates

A feature is commercially complete only when all applicable evidence is tied to the same unchanged current head:

- deterministic focused and full tests, with no skip/xfail/source-rewriting or coverage-denominator tricks concealing a failing owned path;
- realistic simulation-recovery against known truth, reporting at least bias and RMSE and, when interval uncertainty is claimed, empirical coverage under a declared Monte Carlo design and deterministic seed manifest;
- CPU worker-count determinism and CPU/GPU parity for paths that advertise both backends;
- 100% owned production statement/branch coverage and 100% public rustdoc/docstring coverage under the repository contract;
- package/build/install evidence, including installed-wheel tests rather than source-tree import only;
- security/static-analysis/fuzz evidence, dependency integrity, SBOM and build provenance as required by the live protected policy;
- zero valid unresolved review findings and the qualifying independent approval required by the live ruleset;
- protected-branch merge without bypass, stale/predecessor evidence transfer or fabricated gate evidence.

A release additionally requires one integrated protected head with recovery/package/install/reproducibility evidence, rollback instructions, version/changelog coherence, signed/attested distribution evidence where configured, publish success and post-publish verification. PR #1471 (`v0.9.2`) is therefore not current release authority while substantial post-cut product work remains unintegrated.

## 3. Current protected-product gaps

The highest-leverage gaps are listed by product risk rather than by PR age.

| Gap | Current owner evidence | Acceptance before product claim |
| --- | --- | --- |
| Generalized dependence/model specification | #1714 `6abdcc2acab7be463977e35191566119e384c906` | Exact supported/research-candidate/unsupported semantics; no silent local-independence substitution; formulation-specific identification, Rust estimator and recovery before promotion. |
| TEPP temporal boundary | #1716 `f2dcf79b6e903c6a3edb5271ef1861bfcc4da3f8` | Context Map/ADR/PRD/TRD agree that TEPP owns temporal/event composition while fast-mlsirm owns reusable numerical kernels only; no unversioned runtime coupling. |
| Buyer/acquisition execution and CI runner identity | #1717 `9500629c70ca8e5f5e0ed3ef246630534328d73e` | Generic acquisition workflow remains price-neutral; exact-head hosted checks complete; no repository source claim is inferred from organization runner backlog. |
| Static covariance standardization owner contract | #1722 `e1847c07fd7ef8331dcebd0dd588b1381cc1231d` | Exact represented-input admission, scale/permutation invariance and Rust numerical ownership; TEPP may consume only after an immutable released contract exists. |
| Mokken/AISP admission and decision controls | canonical writer #1506 `1146527c61dc0e5c9a1ae6c10e31fdb1f86fa849`; stacked explicit-control child #1724 `c4739d83bf747a699789b72502d7e380c0b0e302` | Preserve package-owned response/result/control hardening; require caller-governed `lower_bound` and `alpha` rather than universal heuristic defaults; reconcile and validate the stack before the sole main-facing Mokken writer advances. |
| Supply-chain release evidence | #1692 `401a23765cc7e9686927f32f5a4ad268ff1b26af` | SBOM and provenance generated from the exact reviewed source/distributions; irreversible release/PyPI sinks depend on required evidence without putting SBOM files in the package-upload set. |
| Internal Rust crate distribution boundary | #1694 `8e6c30912685bc5d8991351ca4dea426d2386bdf` at the most recent inspected lane state | `mlsirm-core` and `fast-mlsirm-py` remain internal/non-publishable Cargo packages unless a separately governed Rust SDK product is approved; PyPI/Maturin remains the external package product. |
| Capability support matrix | #1710 `e50033e00dc392d532a4fa941ce390c8ef4e8dbe` | Versioned machine-readable exact artifact must match the real public `FitConfig`/production-estimator vocabulary; unsupported estimator identities remain unadvertised and fail closed. |
| Multiple-membership/crossed recovery | #1536 `2b5f406c08dd2f807d71ccdf9697857636067b87` | Deterministic known-truth recovery, classification/membership invariants, CPU-worker parity and explicit limits; no claim of longitudinal/event semantics. |
| Interaction-map stack | #1417 `25b9f9908a2d60782412900e932ba50000760448`, child #1457 `94699c5baa2b734ec38ef49d07f0576efd191883`, maintenance #1725 | Parent/child ancestry must be exact; Rust remains sole owner of reconstruction/explained-share arithmetic and input/result provenance. Stale child evidence must not be transferred. |
| Release cut | #1471 | Restack only after upstream dependency/distribution/supply-chain decisions and integrated scientific work settle; regenerate release evidence from the final protected head. |

This table deliberately omits the current #1519 branch head. A document cannot make its own yet-to-be-created commit SHA immutable by embedding a symbolic value such as “this writer branch”. The immutable source identity for the observation being corrected here is the predecessor baseline artifact named at the top of this file (`3a3865f...` / blob `0f514c...`). The final commit that contains this document is obtained from Git history and is not self-declared inside its own payload.

## 4. Point-in-time repository evidence

A GitHub search performed for this 2026-09-02 refresh recorded **49 open pull requests** and **196 open issues**. These counts are an observation, not a live invariant and must be re-fetched before any later merge/release decision. The protected product base observed for this refresh remained `main@45627700c26c29bca150896a9519a9b7426acb56`.

The repository is intentionally carrying many single-writer feature lanes. A large open count is therefore not itself evidence of product failure, but overlapping direct-to-main writers of the same bounded context are a concrete integration risk. The Mokken #1506/#1724 collision is the current example: #1724 has been stacked onto the exact #1506 parent rather than left as a competing main writer. The interaction-map #1417/#1457 chain similarly requires explicit parent-forward reconciliation instead of parallel protected-main edits.

Queued, pending, in-progress, cancelled, skipped, absent and predecessor-head workflow states are all non-passing. They are also not a reason to mutate a clean source head merely to retrigger CI. Runner-less jobs (`runner_id=0`, no steps/checkout SHA) are control-plane evidence and should be advanced through the organization Actions owner path while independent repository lanes continue.

## 5. Scientific evidence model

Simulation/recovery is part of the production contract, not optional research decoration. Each claimed model/estimator should identify:

1. exact data-generating formulation and identification constraints;
2. true parameters and the transformation/alignment used before error calculation;
3. sample-size, item/person/rater/facet/dependence conditions and missingness/design mechanism;
4. deterministic seed manifest and Monte Carlo replicate count;
5. convergence/failure accounting without dropping inconvenient replicates from the denominator;
6. bias and RMSE for relevant parameters, plus interval coverage and interval width when uncertainty is exposed;
7. CPU worker-count reproducibility and advertised CPU/GPU parity;
8. a declared practical acceptance envelope tied to the supported product claim rather than tuned after seeing the result.

For latent spaces and loading structures, rotation/reflection/sign/permutation non-identifiability must be handled explicitly before recovery error is interpreted. For DIF, facets and mixed/multiple-membership extensions, recovery must match the exact formulation rather than borrowing validation from a related base model. CAT/ATA support additionally requires an explicit item-bank, information/selection/exposure/content-constraint contract and end-to-end recovery/operational simulation before a public support claim.

## 6. Context Graph and Enterprise Architecture boundary

`ContextualWisdomLab/context-graph-contracts` is a foreign-owner Shared Kernel for canonical object/authority references, truth status/origin, valid/system time, provenance, Context Assertion, CloudEvents/schema/conformance/admission. `ContextualWisdomLab/enterprise-architecture-core` is the foreign-owner EA Decision Plane. The fast-mlsirm writer reads their live governance and integration state but does not write source or PR state in those repositories while the Context Fabric writer owns them.

Architecture/lifecycle facts that may eventually be projected include released package/crate/API/service identity, backend/toolchain/provider/version, consuming CWL service dependency, lifecycle, risk, ownership, remediation and transformation. Projection must use an **immutable released** context-graph contract/profile with provenance. Open sibling PR heads are not production contract versions.

Estimator values, latent scores, item/person parameters, DIF/fit diagnostics, recovery metrics and scientific-validity evidence stay in measurement/scientific evidence systems. They are not copied into Context Graph or EA as authoritative architecture facts. Cross-service SQL is prohibited; integration uses released contracts/APIs/events.

At this refresh, context-graph-contracts and EA Core were still treated as unreleased read-only dependencies for fast-mlsirm integration purposes. Before any projection is implemented, refetch their current default/protected branches, releases, open stack ancestry, schema/profile/admission version and conformance evidence. Never infer `develop`/`main` transition or stack numbering from an older snapshot.

## 7. Standards and research traceability

The repository should distinguish a published standard from work in revision. The **2014 Standards for Educational and Psychological Testing** remain the published AERA/APA/NCME baseline used for validity, reliability/precision, fairness and intended-score-use evidence; AERA has an active Standards task force in 2026, so a future revision must not be cited as an already-published replacement until formally released.

Supply-chain evidence should track the current approved standards actually used by workflows. As of this refresh, SLSA v1.2 is the approved current SLSA specification, including Build and Source tracks and provenance guidance. SPDX 3.0.1 is a current published SPDX specification; an implementation that emits another declared SPDX version must identify and validate that exact version rather than silently relabeling output.

Representative primary/research sources that anchor existing or planned model contracts include:

- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.
- Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time structural equation modeling with R package ctsem. *Journal of Statistical Software, 77*(5), 1–35. https://doi.org/10.18637/jss.v077.i05
- Jin, I. H., & Jeon, M. (2019). A doubly latent space joint model for local item and person dependence in the analysis of item response data. *Psychometrika, 84*(1), 236–260. https://doi.org/10.1007/s11336-018-9630-0
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5
- Kang, I., & Jeon, M. (2025). Generalized mixed models for item response data with complex dependence structures. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5
- van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of Statistical Software, 20*(11), 1–19. https://doi.org/10.18637/jss.v020.i11
- Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing optimization algorithms for item selection in Mokken scale analysis. *Journal of Classification, 30*(1), 75–99. https://doi.org/10.1007/s00357-013-9122-y

A paper that motivates a family is not evidence that every generalized mixed/dependence combination is identified or recovered. Novel compositions remain research candidates until exact formulation-specific evidence exists.

## 8. Product-gap priorities after current repair lanes

Once the active repair/stack lanes are integrated, the next buyer/scientific work should be selected from protected-main evidence rather than from roadmap wish lists. The durable priorities are:

- close the generalized model-specification-to-estimator gap so the manifest can distinguish executable support from research candidates without family-specific branching;
- expand true-parameter recovery matrices for the supported dependence, mixed/multiple-membership, rater/facet and DIF formulations, including realistic uncertainty and failure accounting;
- finish deterministic CPU/GPU parity for every advertised accelerated kernel and expose backend capability/version evidence without changing estimator semantics;
- finish installed-wheel and release-provenance evidence from one exact integrated head, including SBOM and post-publish verification;
- keep public capability manifests, PRD/TRD/ADR/Context Map/API docs, rustdoc/docstrings, security/operability and changelog synchronized with protected-main code rather than open-PR aspiration;
- publish integration facts to Context Graph/EA only after an immutable released contract/profile exists, and keep scientific result evidence outside EA authority.

## 9. Change boundary

This baseline records gaps and acceptance contracts. It must not become a second source of numerical formulas, a mutable queue dashboard or a substitute for live GitHub/ruleset/release inspection. Every future refresh must pin the protected base and immutable input artifacts it actually observed, mark volatile PR/issue/check counts as point-in-time observations, and avoid self-referential identities that cannot exist until after the document commit is created.
