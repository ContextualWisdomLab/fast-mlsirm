# Product and technical gap baseline

Status: **Non-authoritative point-in-time product-completion inventory**  
Protected-product basis: `main@b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`  
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

The highest-leverage gaps are listed by product risk rather than by PR age. `ACTIVE PR` means the capability is represented only by an open, unmerged pull-request lane at this observation point; it is not protected-product or released authority. `PROTECTED + ACTIVE DRAFT` means a prerequisite slice has landed on protected `main` while the remaining capability is still an open Draft and must earn its own exact-head evidence.

| Gap | Maturity | Current owner evidence | Acceptance before product claim |
| --- | --- | --- | --- |
| Generalized dependence/model specification | ACTIVE PR | #1714 `6abdcc2acab7be463977e35191566119e384c906` | Exact supported/research-candidate/unsupported semantics; no silent local-independence substitution; formulation-specific identification, Rust estimator and recovery before promotion. |
| TEPP temporal boundary | ACTIVE PR | #1716 `6820ae775cbb415def348818ccaa60a7759073bb` | Context Map/ADR/PRD/TRD agree that TEPP owns temporal/event composition while fast-mlsirm owns reusable numerical kernels only; no unversioned runtime coupling. |
| Buyer/acquisition execution and CI runner identity | ACTIVE PR | #1717 `b228d00b639332bd97dc16995210586443772b70` | Generic acquisition workflow remains price-neutral; exact-head hosted checks complete; no repository source claim is inferred from organization runner backlog. |
| Static covariance standardization owner contract | ACTIVE PR | #1722 `338dbb2d25f32b0e201102e7bf73076846fb57b3` | Exact represented-input admission, scale/permutation invariance and Rust numerical ownership; TEPP may consume only after an immutable released contract exists. |
| Mokken/AISP admission and decision controls | ACTIVE PR | canonical writer #1506 `1146527c61dc0e5c9a1ae6c10e31fdb1f86fa849`; stacked explicit-control child #1724 `063878c04cbd1c8182149582e612f246c60f334d` | Preserve package-owned response/result/control hardening; require caller-governed `lower_bound` and `alpha` rather than universal heuristic defaults; reconcile and validate the stack before the sole main-facing Mokken writer advances. |
| Measurement response/item lifecycle | PROTECTED + ACTIVE DRAFT | binary-response contract #1712 merged into protected `main` as `b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`; dynamic-item Draft #1727 `c94678cc8a23293153c18c734ede5077355fa498` now targets that protected tip after an ancestry-only two-parent reconciliation | Keep observed values separate from missing/adjudication state; freeze concrete dynamic items under immutable blueprint/content/provenance evidence; require validated anchors plus linking evidence for cross-version comparability. These contracts do not themselves claim calibration, DIF, information, CAT/ATA selection or linking arithmetic; those numerical claims remain Rust-owned and require formulation-specific recovery. |
| Supply-chain release evidence | ACTIVE PR | #1692 `401a23765cc7e9686927f32f5a4ad268ff1b26af` | SBOM and provenance generated from the exact reviewed source/distributions; irreversible release/PyPI sinks depend on required evidence without putting SBOM files in the package-upload set. |
| Internal Rust crate distribution boundary | ACTIVE PR | #1694 `8e6c30912685bc5d8991351ca4dea426d2386bdf` at the most recent inspected lane state | `mlsirm-core` and `fast-mlsirm-py` remain internal/non-publishable Cargo packages unless a separately governed Rust SDK product is approved; PyPI/Maturin remains the external package product. |
| Capability support matrix | ACTIVE PR | #1710 `2372f44856d9955b6e390840ae75069a62e24841` | Versioned machine-readable exact artifact must match the real public `FitConfig`/production-estimator vocabulary; unsupported estimator identities remain unadvertised and fail closed. |
| Multiple-membership/crossed recovery | ACTIVE PR | #1536 `2b5f406c08dd2f807d71ccdf9697857636067b87` | Deterministic known-truth recovery, classification/membership invariants, CPU-worker parity and explicit limits; no claim of longitudinal/event semantics. |
| Interaction-map stack | ACTIVE PR | #1417 `25b9f9908a2d60782412900e932ba50000760448`; child #1457 `44e445145c324c18fe0bab16f7e455fdd6f6692f`; ancestry repair #1726 `36fb66af554968815ec4cce98281da4cedc3de06` merged into the child | Parent/child ancestry is now exact: #1457 records #1417 as a parent and is ahead by only its intended explained-share delta. Rust remains sole owner of reconstruction/explained-share arithmetic and input/result provenance. #1725 is closed-as-merged/superseded because the #1726 two-parent ancestry repair incorporated the same parent purpose; neither maintenance PR is protected-main product authority. |
| Release cut | ACTIVE PR | #1471 | Restack only after upstream dependency/distribution/supply-chain decisions and integrated scientific work settle; regenerate release evidence from the final protected head. |

This table deliberately omits the current #1519 branch head. A document cannot make its own yet-to-be-created commit SHA immutable by embedding a symbolic value such as “this writer branch”. The immutable source identity for the observation being corrected here is the predecessor baseline artifact named at the top of this file (`3a3865f...` / blob `0f514c...`). The final commit that contains this document is obtained from Git history and is not self-declared inside its own payload.

## 4. Point-in-time repository evidence

A GitHub search performed for this 2026-09-02 refresh recorded **49 open pull requests** and **196 open issues**. These counts are an observation, not a live invariant and must be re-fetched before any later merge/release decision. The protected product base observed for this refresh is now `main@b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`, where #1712 landed through the protected merge path.

The repository is intentionally carrying many single-writer feature lanes. A large open count is therefore not itself evidence of product failure, but overlapping direct-to-main writers of the same bounded context are a concrete integration risk. The Mokken #1506/#1724 collision is the current example: #1724 has been stacked onto the exact #1506 parent rather than left as a competing main writer. The interaction-map #1417/#1457 chain has now completed its ancestry-only reconciliation: #1726 merged a tree-empty two-parent record into the child, leaving #1457 at `44e445145c324c18fe0bab16f7e455fdd6f6692f` with exact parent ancestry. GitHub subsequently recognizes #1725 as merged/superseded because that parent purpose is already contained; it is not an additional source integration. The Measurement root #1712 is no longer an open stack prerequisite: it merged as protected `main@b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`. Draft #1727 now bases on that exact protected tip; current head `c94678cc8a23293153c18c734ede5077355fa498` is a tree-identical ancestry-only reconciliation of predecessor `9f7d31793bace47784faf73b2feae9ef7a0dd20e` with protected main, so predecessor checks/reviews remain non-transferable despite unchanged feature bytes.

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

Fresh 2026-09-02 read-only inventory still reports `develop` as the default branch for both context-graph-contracts and EA Core, and both GitHub release lists are empty. Context Graph's live `develop@99cb5468ba3c15c5e79688f53dee74724fae2d13` remains protected while its open stack explicitly treats protected `main` migration and the first immutable source-bound contract release as prerequisites. EA's open Context Fabric consumer projection likewise remains fail closed on provisional/open CGC identities. fast-mlsirm therefore has no immutable released CGC/EA integration authority to pin yet; do not consume mutable sibling heads or infer the intended transition from older snapshots.

## 7. Standards and research traceability

The repository should distinguish a published standard from work in revision. The **2014 Standards for Educational and Psychological Testing** remain the published AERA/APA/NCME baseline used for validity, reliability/precision, fairness and intended-score-use evidence; AERA still lists an active Standards task force in 2026, so a future revision must not be cited as an already-published replacement until formally released.

Supply-chain evidence should track the current approved standards actually used by workflows. SLSA v1.2 is the current approved SLSA specification, including the Source Track alongside the Build Track. SPDX 3.0.1 remains a published stable 3.x specification surface while SPDX 3.1 is still release-candidate material; an implementation must identify and validate the exact SPDX version it emits rather than silently relabeling output.

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
- complete the Measurement response/item lifecycle so dynamic or generated item evidence can be frozen and compared without conflating adjudication, validation, calibration, anchoring or linking, then add Rust-owned eligibility/calibration/DIF/information/linking kernels only with recovery evidence;
- expand true-parameter recovery matrices for the supported dependence, mixed/multiple-membership, rater/facet and DIF formulations, including realistic uncertainty and failure accounting;
- finish deterministic CPU/GPU parity for every advertised accelerated kernel and expose backend capability/version evidence without changing estimator semantics;
- finish installed-wheel and release-provenance evidence from one exact integrated head, including SBOM and post-publish verification;
- keep public capability manifests, PRD/TRD/ADR/Context Map/API docs, rustdoc/docstrings, security/operability and changelog synchronized with protected-main code rather than open-PR aspiration;
- publish integration facts to Context Graph/EA only after an immutable released contract/profile exists, and keep scientific result evidence outside EA authority.

## 9. Change boundary

This baseline records gaps and acceptance contracts. It must not become a second source of numerical formulas, a mutable queue dashboard or a substitute for live GitHub/ruleset/release inspection. Every future refresh must pin the protected base and immutable input artifacts it actually observed, mark volatile PR/issue/check counts as point-in-time observations, and avoid self-referential identities that cannot exist until after the document commit is created.

## 10. Exact-head commercialization control-plane refresh — 2026-09-02

Fresh live evidence for this writer still observes protected `ContextualWisdomLab/fast-mlsirm` `main@b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`, **49 open pull requests**, and **196 open issues**. Central `ContextualWisdomLab/.github` protected `main@78271917b526469c559fa75cb5ee39426e5494d1` has already integrated the scheduler stale-head cancellation repair (#1669), the obsolete hourly repair-workflow cleanup (#1673), and the Noema deleted/base-side evidence repair (#1564); #1567 is not an independent remaining prerequisite because its valid hollow-CodeGraph cleanup delta was absorbed into #1564.

The current static covariance owner lane is `ContextualWisdomLab/fast-mlsirm#1722@338dbb2d25f32b0e201102e7bf73076846fb57b3`, mergeable against the protected product tip with exactly four changed files in the Rust owner-contract slice. Its substantive numerical review findings are resolved in source/test history, but landing evidence remains incomplete: no qualifying independent approval is present, required `noema-review` is still queued, and the exact-head required CodeQL workflow has produced a `startup_failure` before any job materialized. This is control-plane admission evidence, not a reason to weaken or churn the Rust leaf implementation.

The causal owner lane is `ContextualWisdomLab/.github#1150`. The existing single writer was extended without force-push. Commit `9a187a088a1ae78b95c2eefa522fdbfe6ec38f1a` adds `ContextualWisdomLab/fast-mlsirm` to the explicit bounded read-only Actions queue-health allowlist and updates the corresponding contract test so the owner-plane collector observes the same queued, zero-job `startup_failure`, and cancelled-before-runner states affecting the product lane. A newly materialized review then found that the post-evidence pull-request identity read did not receive the same bounded transient-incompleteness retry as earlier identity reads. RED `5031e0bcb498add8d5833e7ebc0f8400a1835e4f` adds transient-recovery and persistent-failure regressions; GREEN `36639d090fd24c894e06fe39d01bac2dcfa0c4a4` retries that post-evidence identity read once and still fails closed if the retry remains incomplete or invalid. Fresh exact-head protected Checks for this owner lane are non-transferable and must complete on the unchanged GREEN head before merge.

Central Noema repair lane `ContextualWisdomLab/.github#1672@db13a2df02709a20c25e98dce215e5561ee1b53d` remains mergeable and all currently observed review threads were resolved after current-source verification. Its implementation removes the unsupported repository-owned 900-second model-repair deadline and duplicate caller-side model request, fixes Actions model traffic to `orchestrator/free`, keeps deterministic local validation/fail-closed semantics, and leaves provider discovery/structured-output repair/failover/upstream completion with `contextual-orchestrator`. Exact-head required evidence is still non-terminal; predecessor results do not transfer.

Buyer-visible effect: the Rust covariance contract itself is no longer the highest-risk blocker in this slice. Organization Actions admission and central review evidence freshness currently dominate release predictability. The owner repair therefore remains a control-plane observability/correctness lane, while fast-mlsirm keeps the mathematical kernel unchanged until exact-head evidence proves a leaf defect. No bypass, self-approval, stale-evidence promotion, source-copy workaround, mutable sibling-head consumption, or release claim is recorded by this refresh.