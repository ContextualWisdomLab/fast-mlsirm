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
| TEPP temporal ownership boundary | ACTIVE PR | #1716 `91c6563c2a3c4d8bddd75b94d261d92e864cf97e` | PRD/TRD/ADR/Context Map and executable fitness tests must agree that TEPP owns temporal/event composition while this repository owns reusable psychometric numerics only; historical ADRs must carry the same explicit ownership qualification. |
| Acquisition/readiness, hosted-runner identity and GPU merge-gate parity | ACTIVE PR | #1717 `a53033aa949faf6494945eeff83c3f920c7cbf09` | The protected `python` context depends on both the CPU matrix and explicit GPU parity success. Require fresh exact-head CI/security/review evidence after the RED→GREEN workflow and stale-contract repair. |
| Static covariance standardization | ACTIVE PR | #1722 `338dbb2d25f32b0e201102e7bf73076846fb57b3` | Exact represented-input admission, scale/permutation invariance and Rust numerical ownership; TEPP may consume only after an immutable released versioned contract exists. |
| Mokken/AISP admission and decision controls | ACTIVE STACK | canonical #1506 `cd46160c0a035fec2ded13fbacb11159f0d33ad4`; child #1724 `7764245d3d7618de08dc57e1434bb9b8e8c918ac` on that exact parent | Preserve package-owned response/result/control hardening and caller-governed `lower_bound`/`alpha`; the current parent removes a platform `pytest.skip` from longdouble precision evidence while preserving explicit capability classification, and the child is non-force reconciled with merge base exactly equal to the parent. |
| RSM response/result provenance boundary | ACTIVE PR | #1699 `ac9658ebda1d69a86c1ba61ab8ef36a2e346c573` | Keep likelihood/ECM/scoring arithmetic Rust-owned while sealing caller response evidence and the exact PyO3 native-result envelope, including bounded likelihood-trace admission before package-owned copying. |
| Measurement response/item lifecycle | PROTECTED + ACTIVE DRAFT | binary-response contract landed as `main@b5a3a0c...`; dynamic-evaluation Draft #1727 `13848f9d5a089be4f727f2982def507c96ec6fbc` | Keep observed value, nonresponse and adjudication state separate; bind dynamic items to immutable criterion-set identity/provenance; require validated anchors/linking before cross-version comparability. Fresh exact-head review found the production factory had regressed to preserving caller order despite existing criterion-membership and item-set order-invariance RED tests; `6286318c...` and current `13848f9d...` restore canonical membership identities without conflating administration sequence, while the earlier test-contract repair preserves wrong-carrier `TypeError` versus semantically empty exact-item-set rejection. No calibration, DIF, information or CAT/ATA claim follows from the state/identity contracts alone. |
| Supply-chain release evidence | ACTIVE PR | #1692 `873f4bb5fdb5a215d43273c46868ff545bfaf09e` | Exact-source wheels/sdist, SPDX SBOM and builder-local provenance; irreversible PyPI/GitHub release sinks must depend on the required evidence without mixing SBOM into package upload. |
| Rust distribution boundary and `sha2` 0.11 | ACTIVE PR | #1694 `756cf889a111717725de806329e8e5a64bbb5bc0` | `mlsirm-core` and `fast-mlsirm-py` remain internal `publish = false` Cargo packages; PyPI/Maturin remains the external product unless a separately governed Rust SDK is approved. Preserve all lock roots and SHA-256 wire identity. |
| Standalone Cargo dependency governance | ACTIVE PR | #1697 `e90ea988cc2ef3bcca3bfb9eb08f8aa851f3d742` | Dependabot must cover the root, standalone PyO3 and fuzz Cargo lock roots without weakening `--locked` verification; shared dependency updates must not silently leave a production wheel graph stale. |
| Machine-readable capability support matrix | ACTIVE PR | #1710 `0d97c877f1496327f3f86fec641755bed4364438` | Exact 1.0 artifact must match public `FitConfig`/production estimator vocabulary; unsupported estimator identities remain unadvertised and fail closed. |
| Multiple-membership/crossed recovery | ACTIVE PR | #1536 `77bef27cff780b909be484b52be35e97be752780` | Known-truth bias/MAE/RMSE, membership invariants, bounded direct-Rust admission and CPU worker determinism; do not overclaim interval coverage, variance-component recovery or longitudinal semantics. |
| Test-evidence non-execution governance | ACTIVE PR | #1733 `c17462e6ef25153fca30f1ca6accf50b4e029ca6` | Skip/xfail/xpass outcomes cannot make an otherwise-successful required invocation GREEN; preserve stronger pytest exit classifications. Fresh review also proved descriptor-relative atomic-write tests could return normally when a platform primitive was missing; source-level RED `5dc238d...` and GREEN `3ab1ebf...` now make the missing prerequisite failing evidence instead of a false pass. Require exact-head hosted execution before integration. |
| Release cut | ACTIVE PR | #1471 | Restack only after upstream distribution/supply-chain/product decisions settle; regenerate release evidence from the final protected integrated head. |

Fresh GitHub inventory at this observation records **50 open pull requests** and **197 open issues**. Those counts are volatile and must be re-read before later decisions.

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

Fresh live inventory still reports `develop` as the default branch for both repositories. Context Graph `develop@99cb5468ba3c15c5e79688f53dee74724fae2d13` and EA Core `develop@1c0fa8b15ceb9e72186274aeb255d6777eb84ef4` remain the observed protected development tips. Active organization ruleset `18156473` targets `~DEFAULT_BRANCH` and currently requires one approving review, review-thread resolution, central required workflows, deletion protection and non-fast-forward protection. Therefore the accepted protected-`main` transition described by their owner lanes has not yet occurred and must not be inferred from roadmap prose. `.github#1137` remains the administrative owner path.

Context Graph currently has **14 open PRs and 2 open issues**. High-priority issue #24 keeps source-bound immutable release evidence open; Draft #25 is the owner repair. EA Core currently has **24 open PRs and 2 open issues**; its Context Fabric consumer work remains fail closed on provisional CGC identity. Both GitHub release lists are empty. Therefore fast-mlsirm has no immutable released CGC contract/profile to pin yet.

Architecture/lifecycle facts that may later be projected include released package/crate/API/service identity, backend/toolchain/provider/version, consuming CWL dependency, lifecycle, risk, ownership, remediation and transformation. Projection must use a released versioned Context Assertion/CloudEvent/conformance contract with provenance. Estimator values, latent scores, item/person parameters, DIF/fit diagnostics, recovery metrics and scientific-validity evidence remain scientific evidence and must not be copied into EA authoritative architecture truth.

## 6. Exact-head Actions/control-plane state

The dominant landing blocker across otherwise source-repaired fast-mlsirm lanes is currently organization Actions admission rather than a verified numerical defect. The newest substantive Mokken owner head #1506 `cd46160c0a035fec2ded13fbacb11159f0d33ad4` is exactly based on protected `main`, Ready and mergeable. Required CodeQL PR run `33691314629` terminated `startup_failure` with zero jobs; repository CI `33691312917` remains pre-job with `jobs=[]`; Security Scan `33691312901` materialized `scorecard`, `dependency-review`, `osv-scan` and `trivy-fs`, but all remain runnerless with no checkout/source steps. That exact head contains a real portability/test-governance repair rather than a no-op retrigger.

Dependent #1724 was reconciled non-destructively onto that owner head as `7764245d3d7618de08dc57e1434bb9b8e8c918ac`; its merge base is exactly `cd46160...`, `ahead_by=25`, `behind_by=0`. Exact child CI `33691560237` has `jobs=[]`; CodeQL `33691560184` materialized `Analyze (actions)` but it remains queued without a runner. These exact-current results supersede all predecessor-head landing evidence.

The same control-plane class is independently reproduced on #1717 `a53033aa949faf6494945eeff83c3f920c7cbf09`, #1699 `ac9658ebda1d69a86c1ba61ab8ef36a2e346c573`, #1733 `c17462e6ef25153fca30f1ca6accf50b4e029ca6`, and the substantive dynamic-evaluation repair #1727 `13848f9d5a089be4f727f2982def507c96ec6fbc`. On #1733, valid review RED `5dc238d25cd10e6889c9900d49714c793c19e01e` proved the prior atomic-write capability helper still produced false passes; GREEN `3ab1ebf3a3e73632bd46d49998a009dee5f0a728` makes missing descriptor-relative prerequisites fail closed, and current `c17462e6...` records the repair. Its required `CodeQL PR` run `33707819626` terminates `startup_failure` with `jobs=[]`; repository CI `33707818706` is pending with `jobs=[]`; Security Scan `33707818677` has four exact-head `ubuntu-24.04` jobs (`100500685908`, `100500686110`, `100500686382`, `100500686396`) that remain queued with no runner identity or steps. Repository CodeQL `33707818688`, Semgrep `33707818684` and OSV `33707818952` remain queued, while Scorecard `33707818698` remains pending.

On #1727, fresh source inspection invalidated predecessor head `506ed8d6...` because both existing order-invariance RED contracts were again violated; `6286318c...` and current `13848f9d...` restore criterion-membership and item-set canonicalization. On that current exact head, required `CodeQL PR` run `33700547557` is terminal `startup_failure` with `jobs=[]`; repository CI `33700546493` is pending; CodeQL `33700546477`, Security Scan `33700546431`, and OSV `33700546859` are queued, while Semgrep `33700546369` and Scorecard `33700546412` are pending at the latest read. This does not authorize moving otherwise-clean leaf source merely to retrigger.

The canonical central owner path remains `ContextualWisdomLab/.github#712`. Protected `.github/main` has advanced to exact `bf5970df983dd36e3372c124778ec60857414eba`. Queue-health PR #1150 remains exact `bbacf9e81ae954eb8365fbfe1856d8698a768a4a` but still records predecessor base `8c085835fbf77de2321b72fa6b8dd946227e523e`; GitHub now reports it non-mergeable. That central-owner ancestry drift was handed to #712 with RED/GREEN acceptance for a non-force reconciliation or verified canonical successor. Until the central writer repairs it, #1150's predecessor current-main claims are stale and cannot be treated as landing authority.

fast-mlsirm must not compensate by weakening gates, changing clean source merely to retrigger, promoting predecessor success, self-approving, bypassing protection or fabricating evidence. The protected central Noema repair already removes the unsupported caller-side 900-second repair deadline and duplicate model request, keeps Actions model traffic on `orchestrator/free`, and leaves provider discovery/failover/structured-output repair with `contextual-orchestrator`; that remains foreign-owner evidence rather than a fast-mlsirm source responsibility.

## 7. Next executable commercialization priorities

After active lanes clear their exact-head gates, priority should remain evidence-led rather than roadmap-led: connect the generalized Model Specification contract to formulation-specific Rust estimators and recovery; complete the Measurement item lifecycle without conflating response state, validation, calibration, DIF, information and linking; expand realistic recovery matrices for supported dependence/mixed/facet/DIF formulations; close advertised CPU/GPU parity; make installed-wheel and release provenance reproducible from one exact integrated head; and publish Context Graph/EA integration facts only after an immutable released Shared Kernel contract exists.

This baseline must not become a second source of numerical formulas, a mutable queue dashboard or a substitute for live GitHub/ruleset/release inspection. Every refresh must pin the protected base and exact source evidence it actually observed, repair stale stack/head references instead of preserving them as narrative, and never treat queued/predecessor evidence as passing.
