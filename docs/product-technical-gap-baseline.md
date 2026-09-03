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
- zero valid unresolved review findings and the qualifying approval required by the live policy;
- normal protected merge without self-approval, bypass, gate weakening, force update or predecessor-evidence transfer.

Queued, pending, in-progress, cancelled, skipped, absent and `startup_failure` states are non-passing but are not reasons to churn a clean source head. A release additionally requires one exact integrated protected head with recovery, package/install, reproducibility and rollback evidence; coherent version/CHANGELOG/tag state; immutable distribution/SBOM/provenance evidence; publish success; and post-publish verification.

The latest immutable GitHub release remains `v0.9.1` (published 2026-08-26). PR #1471 proposes `v0.9.2`, but it is not release authority while upstream product, dependency and supply-chain lanes remain open.

## 3. Current high-leverage product gaps

`ACTIVE PR` means open/unmerged evidence, not protected-product authority. `ACTIVE DRAFT` means the live owner lane is intentionally non-landing while ownership, dependency or exact-head evidence is being repaired. `PROTECTED + ACTIVE DRAFT` means a prerequisite slice has landed but the remaining capability still requires its own exact-head acceptance.

| Gap | Maturity | Current exact owner evidence | Acceptance before product claim |
| --- | --- | --- | --- |
| Generalized dependence/model specification | ACTIVE DRAFT | #1714 `9a09bbd9cef4a7b8b4669c47be8c48287fc06152` | Preserve supported/research-candidate/unsupported semantics and unique cross-classification-axis identity; require exact equation, identification, Rust estimator and formulation-specific recovery before promotion; reacquire all exact-head evidence after the topology repair. |
| TEPP temporal ownership boundary | ACTIVE PR | #1716 `91c6563c2a3c4d8bddd75b94d261d92e864cf97e` | PRD/TRD/ADR/Context Map and executable fitness tests must agree that TEPP owns temporal/event composition while this repository owns reusable psychometric numerics only. |
| Acquisition/readiness, hosted-runner identity and GPU merge-gate parity | ACTIVE PR | #1717 `a53033aa949faf6494945eeff83c3f920c7cbf09` | Protected `python` evidence must include CPU matrix and explicit GPU parity success on the same current head. |
| Static covariance standardization | ACTIVE PR | #1722 `338dbb2d25f32b0e201102e7bf73076846fb57b3` | Exact represented-input admission, scale/permutation invariance and Rust numerical ownership; TEPP may consume only an immutable released versioned contract. |
| Population-label admission | ACTIVE PR | canonical #1522 `64427a640983e5f849aad16c326697a688b646f6` | Preserve exact callback-free label identity and signed-int64 boundaries; platforms without genuinely wider `longdouble` precision must fail the relevant evidence rather than report a passing skip. |
| Compensatory 2PL response/result admission | ACTIVE PR | canonical #1646 `7adde29f0e1214f539047632536be67f578ce377` | Preserve Rust-owned likelihood/estimation/scoring, package-owned response/result snapshots and exact result envelopes; the branch is non-force restacked on current protected `main` and must regenerate all exact-head gates after that ancestry move. |
| Mokken/AISP admission and decision controls | ACTIVE STACK | canonical #1506 `cd46160c0a035fec2ded13fbacb11159f0d33ad4`; child #1724 `7764245d3d7618de08dc57e1434bb9b8e8c918ac` | Preserve package-owned response/result/control hardening and caller-governed `lower_bound`/`alpha`; integrate parent first and regenerate child evidence after ancestry movement. |
| RSM response/result provenance boundary | ACTIVE PR | #1699 `ac9658ebda1d69a86c1ba61ab8ef36a2e346c573` | Keep likelihood/ECM/scoring arithmetic Rust-owned while sealing caller response evidence and exact PyO3 result envelopes. RSM lossless-tolerance tests remain this lane's ownership, not #1733's. |
| Measurement response/item lifecycle | PROTECTED + ACTIVE DRAFT | binary-response contract landed as `main@b5a3a0c...`; dynamic-evaluation Draft #1727 `6179ca2d7d0a9d24719f9bd70fc8b60698e2b745` | Keep observed value, nonresponse and adjudication state separate; bind dynamic items to immutable criterion-set identity/provenance; require validated anchors/linking before cross-version comparability. Temporal/administration sequence remains TEPP-owned. |
| Supply-chain release evidence | ACTIVE PR | #1692 `873f4bb5fdb5a215d43273c46868ff545bfaf09e` | Exact-source wheels/sdist, SPDX SBOM and builder-local provenance; irreversible publish sinks depend on the evidence. |
| Rust distribution boundary and `sha2` 0.11 | ACTIVE PR | #1694 `756cf889a111717725de806329e8e5a64bbb5bc0` | Internal Cargo packages remain `publish = false`; PyPI/Maturin remains the external product absent a separately governed Rust SDK. |
| Standalone Cargo dependency governance | ACTIVE PR | #1697 `e90ea988cc2ef3bcca3bfb9eb08f8aa851f3d742` | Dependabot and `--locked` verification must cover root, standalone PyO3 and fuzz lock roots without silently leaving the production wheel graph stale. |
| Machine-readable capability support matrix | ACTIVE PR | #1710 `b6336aab9ba1bf2a39acd5a1f13108dc3655d746` | Artifact must match public `FitConfig`/estimator vocabulary; unsupported identities remain unadvertised and fail closed; module entrypoint stays inside owned coverage. |
| Multiple-membership/crossed recovery | ACTIVE PR | #1536 `77bef27cff780b909be484b52be35e97be752780` | Known-truth bias/MAE/RMSE, membership invariants, bounded direct-Rust admission and CPU worker determinism; no unearned interval-coverage or longitudinal claim. |
| Test-evidence non-execution governance | ACTIVE DRAFT | #1733 `07cc43803736500df145f8a597cf1b9f1ef142d4`; canonical overlapping owners #1522 `64427a640983e5f849aad16c326697a688b646f6` and #1646 `7adde29f0e1214f539047632536be67f578ce377` | Skip/xfail/xpass cannot make required evidence GREEN. #1733 remains Draft after single-writer review: population-label and 2PL portability deltas now belong to their canonical PRs and are only mirrored temporarily in #1733 until those owners integrate; Brennan-Kane mastery-cut, WLE and CDM portability remain #1733-owned unless a fresh canonical writer is identified. RSM and Rasch skip sites remain with #1699 and #1516. |
| Diagnostics report accessibility | ACTIVE PR | #1740 `98e23f26dbc3f7210eff31b79da452df761c78ba` | Preserve skip-link normal-text contrast >= 4.5:1 in light/dark themes and existing no-transition behavior; require current-head accessibility/test/security evidence. |
| Item-bank report accessibility | ACTIVE PR | clean successor #1741 `68d4e5344618170310b459c10a90a0c7f6768899` | Preserve focus target/ring, reduced-motion, semantic row headers and tabular numerals; regenerate every landing gate and qualifying review on the successor. |
| Release cut | ACTIVE PR | #1471 | Restack only after upstream distribution/supply-chain/product decisions settle; regenerate release evidence from the final protected integrated head. |

Fresh GitHub inventory at this observation records **53 open pull requests** and **198 open issues**. Those counts are volatile and must be re-read before later decisions.

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

Fresh live inventory still reports `develop` as the default branch for both repositories. Context Graph `develop@99cb5468ba3c15c5e79688f53dee74724fae2d13` and EA Core `develop@1c0fa8b15ceb9e72186274aeb255d6777eb84ef4` remain the observed development tips. Context Graph has **14 open PRs and 3 open issues**; issue #26 is an external-metadata interoperability contract owner lane and does not change fast-mlsirm's scientific ownership. EA Core has **24 open PRs and 3 open issues**; issue #44 is the corresponding governed OpenMetadata projection consumer lane. Open foreign PR heads remain provisional evidence, not released authority.

Context Graph release/source-provenance prerequisite #25 remains Draft on its owner path. EA consumer projection work remains fail closed on provisional Context Graph identity. fast-mlsirm must not pin mutable CGC/EA PR heads or copy estimator values, latent scores, item/person parameters, DIF/fit diagnostics, recovery metrics or scientific-validity evidence into EA authoritative architecture truth.

Architecture/lifecycle facts may be projected only after a released versioned Context Assertion/CloudEvent/conformance contract exists with immutable provenance: package/crate/API/service identity, backend/toolchain/provider/version, consuming CWL dependency, lifecycle, risk, ownership, remediation and transformation. No cross-service SQL or source copy is permitted.

## 6. Live protection and Actions/control-plane state

Protected `main` remains exact `b5a3a0c1057d4b53d7a4bb18e0de69f630c2b45c`. Repository branch protection currently hard-requires status contexts including `Analyze (actions)`, `close-empty`, `scan-pr-queue`, `dependency-review`, `osv-scan`, `trivy-fs`, `scorecard`, `required-workflow-bootstrap`, `coverage-evidence`, `opencode-review`, `python`, `rust`, `package`, and `fuzz`.

Inherited organization ruleset `18156473` is active on `~DEFAULT_BRANCH`, requires one approving review and review-thread resolution, and currently binds **nine** central required workflows: close-empty, OpenCode review, PR review/merge scheduler, security scan, Strix, Semgrep, Noema review, Scorecard and OSV Scanner. Central CodeQL is intentionally absent from that required-workflow list after `.github#1719` established that `github/codeql-action` cannot execute as a ruleset-required workflow.

The bare approving-review count is itself a known central governance defect under the declared solo-maintainer model: there is no named eligible independent reviewer, while self-approval and bot/model-as-human approval remain forbidden. Canonical owner paths `.github#772/#1351` and owner-plane PR `.github#1644` define the scoped repair: remove only the structurally impossible generic approval count and routine bypass while preserving or strengthening deterministic workflow/security/coverage/package/SBOM/provenance, exact-head, thread-resolution, deletion and non-fast-forward controls. Until that central repair is actually applied to the live ruleset, the one-approval requirement remains mechanically binding; fast-mlsirm must not work around it with self-approval, bot approval or administrator bypass.

The central CodeQL correction exposes a separate integration defect in fast-mlsirm: repository branch protection still independently requires `Analyze (actions)`. A fresh substantive model-specification repair gives a current canary on #1714 exact `9a09bbd9cef4a7b8b4669c47be8c48287fc06152`: repository CI run `33759478916` remains pending with `jobs=[]`; CodeQL run `33759478952` has materialized required `Analyze (actions)` job `100661929143` but it remains queued before runner assignment/source execution with `runner_id=null`, empty runner/group identity and `steps=[]`, while sibling `Analyze (python)` job `100661930535` completed `skipped`. OSV, Security Scan, Scorecard and Semgrep are also non-passing on that unchanged head. The exact cross-layer branch-protection/workflow reconciliation and runner-acquisition RED/GREEN acceptance are recorded on canonical central owner path `ContextualWisdomLab/.github#712`.

#1714 remains Draft after its source-level RED → GREEN topology repair. Its source is not moved merely to retrigger hosted execution, and predecessor workflow/review evidence is not transferred. The same rule applies to #1733 and its overlapping portability evidence: #1522 and #1646 must establish their own exact-head evidence, integrate normally, and only then may #1733 drop temporary overlap and regenerate its own landing evidence.

The latest observed protected central `.github/main` is `09ac6366ddd018fd0085368f4b669ba797fd0158` through the required-workflow/code-scanning governance repair. Central owner movement is evidence of control-plane work, not proof that any fast-mlsirm current head is GREEN.

## 7. Next executable commercialization priorities

After active lanes clear exact-head gates, priority remains evidence-led rather than roadmap-led: connect generalized Model Specification contracts to formulation-specific Rust estimators and recovery; complete the Measurement item lifecycle without conflating response state, validation, calibration, DIF, information and linking; expand realistic recovery matrices for supported dependence/mixed/facet/DIF formulations; close advertised CPU/GPU parity; make installed-wheel and release provenance reproducible from one exact integrated head; and publish Context Graph/EA integration facts only after an immutable released Shared Kernel contract exists.

This baseline must not become a second source of numerical formulas, a mutable queue dashboard or a substitute for live GitHub/ruleset/release inspection. Every refresh must pin the protected base and exact source evidence it actually observed, repair stale stack/head references instead of preserving them as narrative, and never treat queued/predecessor evidence as passing.
