# Product and technical gap baseline

Status: **Non-authoritative point-in-time product-completion inventory**  
Protected-product basis: `main@b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`  
Observation date: 2026-09-03

This file is a commercialization and technical-gap inventory, not runtime authority. A capability is authoritative only after its source is integrated into protected `main` and the applicable scientific, package, coverage, security, review, SBOM/provenance and release evidence is terminal success on one unchanged exact head. Open PRs, Drafts, successful predecessor checks and mutable sibling branches are evidence inputs, not product claims.

## 1. Product and ownership boundary

`fast-mlsirm` is the canonical reusable psychometric numerical engine for LSIRM/MLSIRM/MLS2PLM and adjacent dependence/IRT families. Production likelihood, optimization, scoring, information/uncertainty, covariance/correlation, simulation/recovery and other result-affecting vector/linear/matrix arithmetic belong in Rust/PyO3. Python is limited to validation, provenance sealing, marshalling, orchestration, reporting and explicit reference/parity surfaces that do not become a second numerical implementation.

The internal DDD boundary is **Model Specification**, **Estimation**, **Scoring**, **Diagnostics**, **Simulation-Recovery**, **Compute Backend** and **Public Binding**. Cross-context interaction uses explicit contracts rather than implementation imports. TEPP owns temporal/event semantics and composition; fast-mlsirm may own reusable time-indexed psychometric kernels over explicit supplied occasion/time carriers but does not own TEPP clocks, event ontology or temporal workflow semantics. `contextual-orchestrator` owns LLM/provider routing. Scientific/domain truth stays with its canonical owner; cross-service SQL and source copying are prohibited.

Rasch and generic 1PL are not synonyms in product claims. 2PL/3PL/4PL, bifactor, higher-order, two-tier, multifacet/multifactor, cross-loading, DIF, CAT/ATA and generalized dependence support are promotable only for an exact formulation with primary research grounding, identification constraints, deterministic public contract and formulation-specific recovery evidence.

## 2. Commercial merge and release gates

A change is commercially merge-ready only when all applicable evidence refers to the same unchanged exact head:

- deterministic focused and full tests, without skip/xfail/source rewriting or coverage-denominator tricks hiding a failing owned path;
- realistic known-truth simulation/recovery with deterministic seed manifests, convergence/failure accounting, bias and RMSE, and empirical interval coverage when uncertainty is claimed;
- CPU worker-count determinism and CPU/GPU parity for every path that advertises both backends;
- 100% owned production statement/branch coverage and 100% public rustdoc/docstring coverage under the repository contract;
- package/build/install evidence, including installed-wheel execution rather than source-tree import only;
- security/static-analysis/fuzz/dependency evidence plus required SBOM/provenance evidence;
- zero valid unresolved review findings and the qualifying approval required by the live ruleset;
- normal protected merge without self-approval, bypass, gate weakening, force update or predecessor-evidence transfer.

Queued, pending, in-progress, cancelled, skipped, absent and `startup_failure` states are non-passing but are not reasons to churn a clean source head. A release additionally requires one exact integrated protected head with recovery, package/install, reproducibility and rollback evidence; coherent version/CHANGELOG/tag state; immutable distribution/SBOM/provenance evidence; publish success; and post-publish verification.

The latest immutable GitHub release remains `v0.9.1` (published 2026-08-26). PR #1471 proposes `v0.9.2`, but it is not release authority while upstream product, dependency and supply-chain lanes remain open.

## 3. Current high-leverage product gaps

`ACTIVE PR` means open/unmerged evidence, not protected-product authority. `PROTECTED + ACTIVE DRAFT` means a prerequisite slice has landed but the remaining capability still requires its own exact-head acceptance.

| Gap | Maturity | Current exact owner evidence | Acceptance before product claim |
| --- | --- | --- | --- |
| Generalized dependence/model specification | ACTIVE PR | #1714 `92a3f2152033b61ca89661b5ba8a584842e8c3a9` | Preserve supported/research-candidate/unsupported semantics; require exact equation, identification, Rust estimator and formulation-specific recovery before promotion. |
| TEPP temporal ownership boundary | ACTIVE PR | #1716 `6c98b0f1e05bbf4ccf51128f5ed1dd14e9515036` | PRD/TRD/ADR/Context Map and executable fitness tests must agree that TEPP owns temporal/event composition while this repository owns reusable psychometric numerics only. |
| Acquisition/readiness and hosted-runner identity | ACTIVE PR | #1717 `c8621e9f95cc76f26810b0177d6e56e9a1428698` | The non-force two-parent reconciliation contains protected `main@b5a3a0c...` and preserves the branch delta; fresh exact-head checks/reviews are required after the restack. |
| Static covariance standardization | ACTIVE PR | #1722 `338dbb2d25f32b0e201102e7bf73076846fb57b3` | Exact represented-input admission, scale/permutation invariance and Rust numerical ownership; TEPP may consume only after an immutable released versioned contract exists. |
| Mokken/AISP admission and decision controls | ACTIVE STACK | canonical #1506 `2270cb9f53c3fe39710c056d28f78f4aca3e3859`; child #1724 `e8ed9287d04326d9a2794cc14a8a95d96c6f6045` on that exact parent | Preserve package-owned response/result/control hardening; caller-governed `lower_bound` and `alpha`; maintain one main-facing writer and revalidate ancestry after either head moves. |
| Measurement response/item lifecycle | PROTECTED + ACTIVE DRAFT | binary-response contract landed as `main@b5a3a0c...`; dynamic-evaluation Draft #1727 `850c2e28dadea3bc5ae936e88bc47f2ece871c1a` | Keep observed value, nonresponse and adjudication state separate; bind dynamic items to immutable criterion-set identity/provenance; require validated anchors/linking before cross-version comparability. No calibration, DIF, information or CAT/ATA claim follows from the state/identity contracts alone. |
| Supply-chain release evidence | ACTIVE PR | #1692 `873f4bb5fdb5a215d43273c46868ff545bfaf09e` | Exact-source wheels/sdist, SPDX SBOM and builder-local provenance; irreversible PyPI/GitHub release sinks must depend on the required evidence without mixing SBOM into package upload. |
| Rust distribution boundary and `sha2` 0.11 | ACTIVE PR | #1694 `756cf889a111717725de806329e8e5a64bbb5bc0` | `mlsirm-core` and `fast-mlsirm-py` remain internal `publish = false` Cargo packages; PyPI/Maturin remains the external product unless a separately governed Rust SDK is approved. Preserve all lock roots and SHA-256 wire identity. |
| Machine-readable capability support matrix | ACTIVE PR | #1710 `0d97c877f1496327f3f86fec641755bed4364438` | Exact 1.0 artifact must match public `FitConfig`/production estimator vocabulary; unsupported estimator identities remain unadvertised and fail closed. |
| Multiple-membership/crossed recovery | ACTIVE PR | #1536 `77bef27cff780b909be484b52be35e97be752780` | Known-truth bias/MAE/RMSE, membership invariants, bounded direct-Rust admission and CPU worker determinism; do not overclaim interval coverage, variance-component recovery or longitudinal semantics. |
| Release cut | ACTIVE PR | #1471 | Restack only after upstream distribution/supply-chain/product decisions settle; regenerate release evidence from the final protected integrated head. |

Fresh GitHub inventory at this observation records **49 open pull requests** and **196 open issues**. Those counts are volatile and must be re-read before later decisions.

## 4. Scientific acceptance model

Every claimed estimator/model must identify the exact data-generating formulation and identification constraints, true parameters and any alignment/rotation/sign/permutation transform, sample/design/dependence/missingness conditions, deterministic seed manifest and Monte Carlo replicate count, convergence and failed-replicate denominator, and prespecified acceptance thresholds. Bias and RMSE are mandatory for relevant recovered parameters; interval coverage and width are mandatory when uncertainty is exposed. Latent-space and loading recovery must explicitly handle non-identifiability before error is interpreted. DIF, rater/facet and mixed/multiple-membership extensions require evidence for the exact formulation rather than borrowed validation from a neighboring model.

CAT/ATA requires an explicit item-bank, information/selection, exposure/content-constraint and operational simulation contract before a public support claim. Synthetic fixtures alone are unit-test evidence, not commercial scientific validation.

Primary/research anchors include:

- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.
- Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time structural equation modeling with R package ctsem. *Journal of Statistical Software, 77*(5), 1–35. https://doi.org/10.18637/jss.v077.i05
- Jin, I. H., & Jeon, M. (2019). A doubly latent space joint model for local item and person dependence in the analysis of item response data. *Psychometrika, 84*(1), 236–260. https://doi.org/10.1007/s11336-018-9630-0
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5
- Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5
- van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of Statistical Software, 20*(11), 1–19. https://doi.org/10.18637/jss.v020.i11
- Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing optimization algorithms for item selection in Mokken scale analysis. *Journal of Classification, 30*(1), 75–99. https://doi.org/10.1007/s00357-013-9122-y

A paper that motivates a family does not establish every generalized-mixed × dependence composition. Novel combinations remain research candidates until the exact formulation is identified and recovered.

## 5. Context Graph and EA boundary — read only

`ContextualWisdomLab/context-graph-contracts` is the foreign-owner Shared Kernel for canonical object/authority references, truth status/origin, valid/system time, provenance, Context Assertion, CloudEvents/schema/conformance/admission. `ContextualWisdomLab/enterprise-architecture-core` is the foreign-owner EA Decision Plane. This fast-mlsirm writer inventories them but does not mutate their source, refs or PR state while the Context Fabric writer owns them.

Fresh live inventory still reports `develop` as the default branch for both repositories. Context Graph `develop@99cb5468ba3c15c5e79688f53dee74724fae2d13` and EA Core `develop@1c0fa8b15ceb9e72186274aeb255d6777eb84ef4` remain the observed protected development tips. Active organization ruleset `18156473` targets `~DEFAULT_BRANCH`; therefore the accepted protected-`main` transition described by their owner lanes has not yet occurred and must not be inferred from roadmap prose. `.github#1137` remains the administrative owner path.

Context Graph currently has **14 open PRs and 2 open issues**. High-priority issue #24 keeps source-bound immutable release evidence open; Draft #25 is the owner repair. EA Core currently has **24 open PRs and 2 open issues**; its Context Fabric consumer work remains fail closed on provisional CGC identity. Both GitHub release lists are empty. Therefore fast-mlsirm has no immutable released CGC contract/profile to pin yet.

Architecture/lifecycle facts that may later be projected include released package/crate/API/service identity, backend/toolchain/provider/version, consuming CWL dependency, lifecycle, risk, ownership, remediation and transformation. Projection must use a released versioned Context Assertion/CloudEvent/conformance contract with provenance. Estimator values, latent scores, item/person parameters, DIF/fit diagnostics, recovery metrics and scientific-validity evidence remain scientific evidence and must not be copied into EA authoritative architecture truth.

## 6. Exact-head Actions/control-plane state

The dominant landing blocker across otherwise source-ready fast-mlsirm lanes is currently organization Actions admission rather than a verified numerical defect. On #1722 exact `338dbb...`, organization-required CodeQL PR terminated `startup_failure` before job materialization while CI/security/Scorecard/CodeQL/fuzz/Semgrep/OSV lanes remained non-terminal; substantive numerical review threads are resolved, but no qualifying approval exists. #1729 exact `7faa16037a3e8697e640bfbac780709ee5297d1f` independently reproduced both control-plane classes: required CodeQL PR `startup_failure` with zero jobs, and sibling jobs materialized with `runner_id=0`, no steps and no checkout.

The canonical central owner path remains `ContextualWisdomLab/.github#712` plus PR #1150. Protected `.github/main` is now `8c085835fbf77de2321b72fa6b8dd946227e523e`. This run reconciled #1150 non-destructively onto that protected tip: current exact head `bbacf9e81ae954eb8365fbfe1856d8698a768a4a` has queue-health predecessor `7d80a06c3a48f6411a17aa41e48b1f7064c5e36a` and protected `main@8c085835...` as parents and compares `behind_by=0`. The effective queue-health implementation/config/docs/tests are preserved without a force update or destructive rebase.

Fresh exact-head central runs on `bbacf9e8...` have materialized but remain non-passing: Noema token-lifetime, Secret Scan, Semgrep, CodeQL PR, Python Security, Scorecard, Security Scan, SBOM and OSV are queued. Security Scan run `33655230050` has four exact-head jobs (`dependency-review`, `trivy-fs`, `osv-scan`, `scorecard`) with `steps: []`, label `ubuntu-24.04`, `runner_id=0` and no runner identity. This is fresh central owner evidence of the same pre-checkout acquisition class; predecessor results do not transfer. fast-mlsirm must not compensate by weakening gates, changing clean source merely to retrigger, or promoting stale success.

The protected central Noema repair already removes the unsupported caller-side 900-second repair deadline and duplicate model request, keeps Actions model traffic on `orchestrator/free`, and leaves provider discovery/failover/structured-output repair with `contextual-orchestrator`. That is protected central owner evidence rather than a fast-mlsirm source responsibility.

## 7. Next executable commercialization priorities

After active lanes clear their exact-head gates, priority should remain evidence-led rather than roadmap-led: connect the generalized Model Specification contract to formulation-specific Rust estimators and recovery; complete the Measurement item lifecycle without conflating response state, validation, calibration, DIF, information and linking; expand realistic recovery matrices for supported dependence/mixed/facet/DIF formulations; close advertised CPU/GPU parity; make installed-wheel and release provenance reproducible from one exact integrated head; and publish Context Graph/EA integration facts only after an immutable released Shared Kernel contract exists.

This baseline must not become a second source of numerical formulas, a mutable queue dashboard or a substitute for live GitHub/ruleset/release inspection. Every refresh must pin the protected base and exact source evidence it actually observed, repair stale stack/head references instead of preserving them as narrative, and never treat queued/predecessor evidence as passing.