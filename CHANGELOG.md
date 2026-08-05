# Changelog

## Unreleased


<!-- BEGIN AUTHORITATIVE CHANGELOG FRAGMENTS -->
### Added

#### Deterministic enterprise explicit-value parser

- Added a provider-neutral deterministic parser for verified explicit calendar
  dates, marked deadlines, allowlisted currency amounts, recurrence frequencies,
  and labeled customer or account identifiers under
  `fast_mlsirm.scoring.enterprise_issue`.
- Added exact source fingerprint and Python Unicode-code-point-count replay,
  UTF-8 span fingerprints, deterministic deadline precedence, overlap rejection,
  bounded output, and hashed identifier payloads without retaining source text or
  clear-text customer identifiers.
- Added fail-closed custom-parser output rebinding, callback redaction, exact span
  verification, duplicate/overlap rejection, and strict explicit-value metadata.
  Provider-owned records are reconstructed as fresh canonical instances, manual
  offsets share the enterprise source-character bound, and deterministic candidate
  producers stop at the configured limit plus one rather than exhausting
  unexpectedly prolific iterators.
- Added immutable content-addressed records that compile exact occurrences into
  the existing directly stated `EvidenceSpanRecord` boundary without adding a
  parallel scoring, observation, result, or engine schema.
- Added deterministic, protocol, privacy, security, exact-decimal, offset,
  ordering-invariance, metamorphic sentiment-independence, and fail-closed tests,
  plus APA 7th standards traceability and conservative interpretation limits.

#### Enterprise issue evidence contracts

- Added immutable content-addressed enterprise source, evidence-span, atomic-issue,
  counterevidence, stakeholder-perspective, and candidate-intervention contracts.
- Added deterministic compilation into the shared scoring `EvidenceReference`
  boundary while preserving facts, inferences, counterevidence, ambiguities, and
  stakeholder value judgments as distinct epistemic roles.
- Added fail-closed provenance, source-revision, sensitive-metadata, and ordering
  tests for the first issue #404 domain-adapter slice.

#### Enterprise issue scoring request compiler

- Added deterministic compilation from accepted enterprise issue, evidence,
  counterevidence, stakeholder-perspective, and candidate-intervention records
  into the existing shared criterion-level `ScoringRequest` contract.
- Added package-managed exact provenance for source revisions, evidence spans,
  epistemic assertion kinds, perspectives, interventions, task revision, rubric,
  assessment, and engine authorization without retaining raw enterprise text.
- Preserved the existing `ScoringEngine`, `ScoreObservation`, and
  `ScoringResult` execution boundaries without adding a parallel result schema
  or sentiment, calibration, ranking, utility, causal, or routing arithmetic.
- Added fail-closed duplicate-evidence, cross-issue replay, unbound perspective
  source, reserved metadata, sensitive-content, ordering-invariance, and shared
  contract delegation tests for issue #404.

#### Accessible standalone essay facets-calibration artifacts

- Added `render_essay_facets_calibration_report_html`, which replay-verifies one governed `EssayFacetsCalibrationReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, design, assessment, rubric, construct, occasion, criterion, respondent, task-revision, rater-engine, category, estimate, convergence, connectedness, iteration, and review-trigger evidence through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- A restrictive meta-delivered Content Security Policy and output encoding reduce injection impact; convergence and connectedness remain integrity prerequisites and do not establish model fit, reliability, fairness, scorer interchangeability, construct validity, global optimality, or deployment authorization.

#### Governed essay many-facet calibration reports

- Added source-text-free, provenance-bound essay many-facet calibration reports that preserve exact criterion, respondent, task-revision, engine, category, Rust-fit, convergence, connectedness, and iteration evidence without cross-criterion averaging or validity claims.
- Added fail-closed shape, finite-value, likelihood-trace, parameter-count, connectedness, and design-fingerprint replay checks plus non-suppressible human-review routing for non-converged or disconnected fits.
- Added complete public documentation, APA 7th equation-to-source traceability, privacy guarantees, deterministic fixtures, Rust-delegation tests, and statement/branch coverage for the new reporting boundary.

#### Accessible standalone essay score report artifacts

- Added `render_essay_score_report_html`, which replay-verifies one governed `EssayScoreReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, assessment, rubric, task-revision, engine, request, result, observation, criterion, trigger, and evidence-reference identities through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- Empty states use explicit status semantics, numeric table cells use tabular numerals, skip links remain visible for any received focus, and motion-sensitive users receive a reduced-motion override without dimming non-hovered report rows.
- A restrictive meta-delivered Content Security Policy and output encoding reduce content-injection impact. Review routing remains an audit signal only and does not establish scoring validity, fairness, reliability, interchangeability, accessibility conformance, security certification, or authorization for consequential deployment.

#### Provenance-bound essay score reports

- Added a provider-neutral, content-addressed `EssayScoreReport` adapter over the existing governed essay request, shared scoring result, and engine descriptor.
- Report construction replays exact request, engine, assessment, rubric, construct, granularity, and criterion provenance before emission.
- Submission review flags, terminal observations, and scored observations without evidence now produce non-suppressible transparent human-review triggers. The report preserves criteria separately and explicitly does not treat absence of a trigger as validity or deployment evidence.

#### Accessible standalone essay validation evidence artifacts

- Added `render_essay_validation_evidence_report_html`, which revalidates one governed `EssayValidationEvidenceReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, assessment, construct, rubric, validation-dataset, automated-engine, human-reference, metric, review-trigger, and interpretation-boundary values through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- A restrictive meta-delivered Content Security Policy and output encoding reduce injection impact; the artifact deliberately excludes score-label vectors, universal thresholds, Boolean pass fields, validity, fairness, model-selection claims, and deployment authorization.

#### Governed automated-essay validation evidence reports

- Added factory-sealed, criterion-specific automated-essay validation evidence reports that bind exact shared assessment, construct, rubric, validation-policy, dataset, automated-engine, and human-reference identities.
- Reused the existing Rust agreement kernel for quadratic-weighted kappa, exact/adjacent agreement, descriptive Pearson association, standardized mean difference, optional human–human degradation, and optional subgroup evidence while deliberately discarding legacy threshold and pass fields.
- Added non-suppressible human-validation and interpretation boundaries, missing-comparator review routing, source-text-free deterministic reports, complete public documentation, APA 7th equation-to-source traceability, Rust-delegation tests, and statement/branch coverage.

### Changed

#### Release cut 0.7.0

- Project version is bumped to 0.7.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the governed automated-essay-scoring adapters, the
  governed criterion facets calibration handoff into the Rust-backed
  many-facet estimator, and the keyboard-scrollable report export
  accessibility fix) now form the `[0.7.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

#### Exact task-revision identity for scoring calibration

- Scoring-request wire schema `1.1` now requires an exact provider-neutral `task_revision_fingerprint` in addition to the logical task identifier. The fingerprint participates in request identity, is propagated by the essay adapter from the complete prompt fingerprint, and prevents changed task content from being silently pooled under one request or calibration item.
- Criterion-level many-facet handoffs now use exact task revisions as the Rust estimator item axis while retaining aligned logical task and task-family labels for audit. Duplicate cells, support, resource bounds, respondent–item connectedness, item–rater connectedness, and response provenance are all revision-indexed; one revision cannot be rebound to a different logical task or family.
- Added an explicit, fail-closed schema-`1.0` request migration that verifies canonical content, fingerprint, public handle, and the authoritative engine-policy projection; requires a caller-supplied task revision; preserves normalized caller metadata; and intentionally does not migrate legacy observations or results. Content identity prevents accidental pooling but does not establish cross-revision comparability, which still requires anchors, invariance/DIF, drift, and recovery evidence.

### Security

#### Descriptor-safe bounded JSON input for automation scripts

- Consolidated governed automation JSON readers behind a descriptor-safe shared
  loader with a 32 MiB inclusive byte bound and a non-recursive 128-level depth
  bound.
- Rejected symbolic links, FIFOs, directories, path replacement, invalid UTF-8,
  malformed JSON, non-object roots, oversized input, and excessive nesting with
  deterministic tests.
<!-- END AUTHORITATIVE CHANGELOG FRAGMENTS -->
## [0.7.0] - 2026-08-04

### Added

#### Governed automated-essay scoring adapters

- Added `fast_mlsirm.scoring.essay` domain adapters that bind exact prompt, submission, evidence-span, assessment, rubric, task-family, criterion, and engine provenance to the existing provider-neutral scoring contracts without storing raw prompt, essay, or source text.
- Added factory-sealed, content-addressed prompt, submission, evidence, and request artifacts; deterministic review flags; bounded source-text-free offsets; replay and source-identity checks; shared engine-policy authorization; and deterministic human/automated fixture coverage.
- The adapter introduces no parallel rubric, observation, result, or provider schema and makes no claim of scoring accuracy, reliability, fairness, scoreability, construct validity, or readiness for consequential automation.

#### Governed criterion-level facets calibration handoff

- Added factory-sealed, content-addressed scoring-facets rating, design, and calibration-bundle contracts that project exact governed scoring requests, results, engines, respondents, tasks, criteria, terminal states, score scales, engine fingerprints, response IDs, and response-content revisions into criterion-specific respondent-by-task-by-rater designs.
- Added fail-closed replay, duplicate, provenance, respondent-task response-revision, observed-support, category-identification, dense-allocation, respondent-task connectedness, and task-rater connectedness gates. Sparse pilot designs may be assembled and audited after at least two categories are observed among scored records, but every declared category must be observed and the design must be connected before it can reach the Rust estimator. Abstained, failed, excluded, and absent cells remain missing and never satisfy category coverage or become low scores.
- Added `fit_scoring_facets_bundle`, which delegates likelihood, EM, quadrature, and parameter updates to the existing Rust-backed `fit_facets` implementation and calibrates analytic criteria separately instead of averaging them. The baseline models respondent or system-run proficiency, task difficulty, common thresholds, and rater severity; it makes no convergence, fit, reliability, fairness, scoreability, construct-validity, rater-interchangeability, causal-utility, or high-stakes automation claim.

### Changed

#### Release cut 0.6.0

- Project version is bumped to 0.6.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the shared provider-neutral assessment and policy
  contracts with their fail-closed canonicalization and callback-redaction
  hardening, and the governed scoring observations with the
  runtime-checkable engine protocol and authoritative assessment replay)
  now form the `[0.6.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

### Fixed

#### Scrollable Export Accessibility

- Added keyboard navigation (`tabindex="0"`, `role="region"`) and visual focus styling (`:focus-visible`) to scrollable exact-value export `<pre>` regions in standalone HTML diagnostics reports.

## [0.6.0] - 2026-08-04

### Added

#### Shared automated-scoring assessment contracts

- Added the provider-neutral `fast_mlsirm.scoring` namespace with immutable construct, engine, calibration, validation, adjudication, monitoring, reporting, and factory-built assessment contracts.
- Assessment artifacts bind exact `RubricSpecification` fingerprints without duplicating rubric levels, own an independent scoring wire-schema version, and expose deterministic full SHA-256 fingerprints plus descriptive 128-bit public handles.
- Bounded canonical metadata is deeply immutable, normalizes equivalent floating values, rejects response or source content, and returns structured non-reflective validation errors. This contract layer performs no psychometric arithmetic and does not itself establish scoring accuracy, reliability, fairness, scoreability, or validity.

#### Governed scoring observations and engine protocol

- The provider-neutral `fast_mlsirm.scoring` core now defines content-addressed human and automated `EngineDescriptor` values, exact `ScoringRequest` bindings, source-text-free `EvidenceReference` provenance, scored/abstained/failed/excluded `ScoreObservation` states, complete `ScoringResult` execution records, and a runtime-checkable `ScoringEngine` protocol.
- Requests bind exact assessment and rubric fingerprints, declared task families, response granularity, criterion sets, allowed rubric scores, response-content digests, and bounded content statistics without retaining raw response text. Results fail closed on missing or duplicate criterion coverage, request/engine mismatches, fabricated scores, missing terminal reasons, duplicate evidence, and mixed holistic/criterion observations.
- A deterministic offline `StaticFixtureEngine` exercises the same public contracts for tests and documentation only. The shared core adds no hosted-provider SDK, network call, credential handling, scoring inference, psychometric arithmetic, or claim of reliability, fairness, model fit, scoreability, or validity.

### Changed

#### Release cut 0.5.0

- Project version is bumped to 0.5.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the testlet, observed-score DIF, and one-facet
  G-theory pilot-calibration handoffs completing the issue #407 handoff
  family, plus the changelog render-parity gate and the restored 0.4.0
  release history) now form the `[0.5.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.5.0] - 2026-08-04

### Added

#### Governed automated-scoring assessment contracts

- Added provider-neutral, factory-sealed, content-addressed `AutomatedScoringAssessmentSpec` artifacts that bind exactly one shared assessment and rubric revision without duplicating criteria or levels.
- Added explicit rubric-template ownership metadata, strict criterion alignment, implementation-neutral submission/evidence requirements, ordered quality-control stages, intended-use and prohibited-use boundaries, mandatory human-adjudication and validation flags, engine-agnostic calibration/validation/monitoring/reporting requirements, subgroup-analysis plans, residual diagnostics, audit-trail retention, and fallback requirements.
- Added deterministic fixtures that instantiate the governed specification for a complete LLM-assisted essay-scoring workflow while preserving provider neutrality and making no claim of readiness for high-stakes use.

#### Governed automated-scoring score reports

- Added provider-neutral, factory-sealed, content-addressed `AutomatedScoringScoreReport` artifacts that bind exactly one shared assessment, rubric revision, task revision, submission fingerprint, engine fingerprint, request fingerprint, and scoring-result fingerprint.
- Added criterion-level score, terminal-state, review-trigger, and evidence-reference projections; fail-closed provenance replay; deterministic report identity; source-text-free serialization; immutable report payloads; and explicit human-review routing.
- Added tests for deterministic replay, criterion coverage, terminal observations, submission-level review flags, evidence expectations, malformed fabricated reports, and serialization privacy.
- The score report is an audit artifact only and does not establish scorer accuracy, reliability, fairness, scoreability, construct validity, causal validity, or authorization for consequential deployment.

#### Accessible automated-scoring score report artifact

- Added deterministic, source-text-free, script-free standalone HTML rendering for one validated `AutomatedScoringScoreReport`, including semantic landmarks, exact-value identity tables, criterion-level score or terminal-state projections, review-trigger tables, evidence-reference tables, and canonical JSON.
- Added a restrictive meta-delivered Content Security Policy, output encoding, keyboard-visible focus, tabular numerals, and reduced-motion behavior without claiming accessibility conformance or security certification.

#### Validation evidence reports for automated scoring

- Added provider-neutral, factory-sealed, criterion-specific validation-evidence reports that bind exact assessment, rubric, construct, validation-policy, dataset, automated-engine, and human-reference identities.
- Added exact/adjacent agreement, quadratic-weighted kappa, descriptive Pearson association, standardized mean difference, optional human–human degradation, and optional subgroup evidence while preserving missing-comparator states and routing incomplete evidence to human review.
- Reused the existing Rust agreement kernel for all metric arithmetic and discarded legacy threshold/pass fields, so reports expose evidence rather than universal acceptance decisions.
- Added deterministic, source-text-free serialization; fail-closed factory construction; complete public documentation; APA 7th equation-to-source traceability; Rust-delegation tests; and statement/branch coverage.

#### Accessible automated-scoring validation evidence artifact

- Added deterministic, source-text-free, script-free standalone HTML rendering for one validated `AutomatedScoringValidationEvidenceReport`, including semantic landmarks, exact-value identity tables, metric tables, subgroup tables, review-trigger tables, interpretation-boundary tables, and canonical JSON.
- Added a restrictive meta-delivered Content Security Policy, output encoding, keyboard-visible focus, tabular numerals, and reduced-motion behavior; artifacts deliberately omit score-label vectors, universal thresholds, pass/fail fields, validity/fairness/model-selection conclusions, and deployment authorization.

### Changed

#### Release cut 0.4.0

- Project version is bumped to 0.4.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the rubric blueprint compiler, privacy manifest,
  screening records, artificial-crowd pilot ratings, Rust partial-credit
  calibration handoff, and the accepted-release item-bank admission gate)
  now form the `[0.4.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.4.0] - 2026-08-04

### Added

#### Blueprint-to-item-generation handoff

- Added provider-neutral generation requests that bind exact blueprint, rubric,
  construct, task-family, slot, prompt, and generator identities without calling
  a hosted provider or retaining generated item content.
- Added bounded generation batches and item-candidate contracts with deterministic
  fingerprints, canonical item-content digests, language and generation metadata,
  provenance replay, duplicate-content rejection, and strict slot/rubric alignment.
- Added exact criterion-level quality-control requirement coverage for every
  candidate plus explicit human-review and prohibition boundaries.

#### Artificial-crowd pilot calibration handoff

- Added immutable, content-addressed pilot rating and calibration-bundle contracts
  that bind accepted screened candidates, task revisions, respondents, raters,
  categories, and sparse missingness while rejecting duplicate or conflicting
  respondent-item-rater cells.
- Added connectedness and estimability preconditions, deterministic dense-array
  allocation, abstention/missing-value preservation, and Rust delegation into the
  existing partial-credit MMLE kernel without duplicating likelihood, gradient,
  Hessian, optimizer, or scoring arithmetic in Python.
- Added deterministic true-parameter recovery and thin Python-delegation tests;
  category-threshold interpretation remains conservative pending stronger
  item-level calibration and validation evidence.

#### Privacy-safe item bank admission gate

- Added fail-closed `ItemBankEntry` and `ItemBankAdmissionRecord` contracts that bind
  exact generation, screening, calibration, retention-policy, assessment,
  rubric, construct, task-family, blueprint-slot, item, and release identities.
- Admission requires an accepted screening decision and an accepted calibration
  release, while preserving source-text-free content and evidence fingerprints,
  temporal lifecycle state, opaque identifiers, and explicit human-review status.
- Added fail-closed replay, cross-family swap, release, calibration, privacy,
  lifecycle, tombstone, deterministic-order, and immutability tests.

#### Quality-control screening records

- Added provider-neutral human and automated screening records for item candidates,
  with criterion-level decisions, evidence fingerprints, reviewer identity,
  deterministic content-addressed records, and explicit human-review routing.
- Added a governed screening-batch decision boundary that verifies exact blueprint,
  rubric, construct, task-family, slot, candidate, and generation-request provenance;
  requires the complete criterion set; rejects duplicate, missing, conflicting, or
  stale records; and prevents automated reviewers from producing final acceptance.
- Added deterministic, privacy-preserving, immutable, provenance-swap, and fail-closed
  tests without embedding a hosted model provider or weakening human governance.

#### Rubric-to-blueprint compiler

- Added immutable `BlueprintSlot` and `AssessmentBlueprint` contracts plus
  deterministic `compile_assessment_blueprint(...)` under `fast_mlsirm.rubric`.
- The compiler validates the canonical rubric and construct graph, emits every
  `(task_family, criterion)` pair exactly once, preserves criterion order from the
  rubric, derives deterministic coverage weights, binds exact criterion and
  task-family fingerprints, and rejects unsupported task families.
- Added fail-closed replay, cross-rubric and cross-construct mismatch, duplicate and
  missing coverage, ordering-invariance, immutability, and nested construct tests.
- Updated the public API, documentation, examples, CHANGELOG, and APA 7th
  traceability for the governed `Rubric → Blueprint → Generation → Screening →
  Artificial Crowd → Rust Calibration → Item Bank` workflow.

### Changed

#### Release cut 0.3.0

- Project version is bumped to 0.3.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the native GPU parity gate, parameter-count helper, rotated
  loading alignment, and benchmark-regression evidence) now form the
  `[0.3.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.3.0] - 2026-08-04

### Added

#### Native GPU backend parity gate

- Added the feature-gated `native_gpu_available()` PyO3 endpoint backed by an
  actual `wgpu` adapter probe plus dispatch telemetry counters for psychometric
  GPU entrypoints.
- Added `require_native_gpu=True` to the public GPU scoring and partial-credit
  wrappers so release validation can fail closed when a real adapter is absent,
  rather than accepting CPU fallback as proof of a native GPU path.
- Added deterministic, GPU-free unit tests for the new requirement and telemetry
  boundary plus an opt-in live-adapter parity test that proves actual native GPU
  dispatch before comparing CPU and GPU outputs.
- Documented the support contract, runtime failure semantics, and release gate:
  simulation or CPU fallback does not count as native GPU evidence.

#### Model parameter count helper

- Added `parameter_count` to `MLSIRM`, which reports the number of free model
  parameters represented by the returned parameter bundle.
- The helper is intended for information criteria and model comparison. It is
  implemented as model-structure bookkeeping and does not make a fit or model
  selection claim.

#### Rust-backed rotated loading alignment

- Added `align_rotated_loadings` and `rotated_loading_distance` to the public
  factor-analysis boundary.
- The implementation performs deterministic exhaustive permutation and sign
  matching in Rust, with a documented `2^k k!` resource bound and fail-closed
  factor-count limit; Python remains a thin validation/marshalling layer.
- Added signed-permutation invariance, reconstruction, malformed-input,
  resource-bound, deterministic, Rust-delegation, and composed-PyO3 registration
  tests.

#### Benchmark regression evidence

- Added opt-in benchmark-regression tests that compare current scaling against
  historical baselines for JL warm starts, joint likelihood evaluation, and
  partial-credit MMLE fitting, with nonzero tolerances and conservative
  interpretation.
- Documented measurement commands, environment metadata requirements, and the
  distinction between regression evidence and universal performance claims.

### Changed

#### Release cut 0.2.0

- Project version is bumped to 0.2.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the Rust MMLE path, safeguarded Newton updates,
  missing-response support, GPU missing-mask support, item scoring, and
  rubric-to-scoring handoff) now form the `[0.2.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.2.0] - 2026-08-04

### Added

#### Rust-backed marginal maximum-likelihood fitting

- Added Rust-backed marginal maximum-likelihood (`fit_mmle`) for MLSIRM with
  deterministic Gauss-Hermite quadrature, log-sum-exp marginalization, safeguarded
  Newton updates, fixed offset/location identification, damped latent-position
  updates, and explicit convergence diagnostics.
- Added true-parameter recovery tests under complete and missing-at-random response
  designs with nonzero tolerances and aligned latent-space RMSE diagnostics.
- Added stress tests for extreme predictors, all-missing rows and columns,
  deterministic replay, and monotone accepted-objective traces.
- Added equation-to-source traceability against Bock and Aitkin (1981), Bock,
  Gibbons, and Muraki (1988), and Gollini and Murphy (2016).

#### Rust-backed safeguarded Newton item updates

- Added Rust-backed safeguarded Newton updates for MLSIRM item intercepts and
  log-discriminations, including analytic gradients and Hessians, finite-value
  checks, positive-definiteness repair, bounded parameter steps, and monotone
  backtracking acceptance.
- Added finite-difference gradient tests, negative-curvature repair tests,
  deterministic recovery fixtures, and Python/Rust parity coverage.

#### Missing-response support

- Added missing-response support across simulation, fitting, diagnostics,
  recovery, and GPU preparation.
- Public validation accepts `NaN` as missing, excludes missing cells from objectives
  and gradients, and rejects infinities or other non-finite inputs.
- Person/item rows with no observations remain finite through deterministic fallback
  initializations.

#### GPU missing-mask support

- Added missing-response masks to GPU likelihood and gradient paths so observed
  entries preserve CPU/GPU parity while missing entries contribute zero.
- Empty observed sets return a zero objective and zero gradients.

#### Rust-backed item scoring

- Added Rust-backed item information and posterior expected-a-posteriori (EAP)
  scoring for 2PL and 3PL items.
- Added thin Python wrappers, deterministic quadrature controls, malformed-input
  validation, Rust-delegation tests, and known-value/finite-difference parity
  coverage.

#### Rubric-to-scoring handoff

- Added `RubricScoringPlan` plus deterministic compilation from an accepted
  `RubricSpecification` and `ConstructGraph` into an external scoring contract.
- The handoff preserves exact rubric, construct, criterion, level, task-family,
  fairness, validation, versioning, and source fingerprints while explicitly
  excluding scoring inference and psychometric arithmetic.

### Changed

#### Release cut 0.1.2

- Project version is bumped to 0.1.2 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the Rust many-facet partial-credit calibration path,
  factor-rotation criteria/scoreability/model-comparison additions, and runtime
  hardening) now form the `[0.1.2] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.1.2] - 2026-08-04

### Added

#### Rust-backed many-facet partial-credit calibration

- Added a Rust-first many-facet partial-credit MMLE path for persons, items,
  common category thresholds, and rater severities with deterministic
  Gauss-Hermite quadrature and baseline-facet identification.
- Added thin Python orchestration, recovery diagnostics, missing-response support,
  exact input validation, Rust/Python delegation tests, and bounded convergence
  reporting.
- Added equation-to-source traceability against Linacre (1989), Engelhard (2013),
  and Eckes (2015); no convergence, fit, reliability, fairness, scoreability,
  construct validity, rater interchangeability, or high-stakes automation claim
  follows from successful optimization alone.

#### Bifactor scoreability diagnostics

- Added Rust-backed bifactor model-based reliability and scoreability diagnostics:
  total-score omega, hierarchical omega for the general factor, factor-specific
  omega after partialling the general factor, explained common variance, and
  percent uncontaminated correlations.
- Added a documented input contract, complete API docstrings, deterministic
  known-value and invariance tests, malformed-input coverage, and APA 7th
  equation-to-source traceability.
- Interpretation remains deliberately conservative: diagnostics are descriptive
  conditional on the fitted loading/error model and do not establish score
  validity, subgroup fairness, causal effects, or readiness for consequential use.

#### Factor-rotation criterion diagnostics

- Added Rust-backed varimax, quartimax, equamax, parsimax, and Crawford-Ferguson
  criterion values with analytic gradients, invariant input validation,
  finite-difference verification, and deterministic known-value tests.
- Clarified that the criteria are descriptive objective values rather than a
  universal rule for selecting the substantively correct rotation.

#### Multistart factor-rotation search

- Added deterministic orthogonal multistart rotation over a caller-supplied
  objective callback, preserving finite-start limitations and returning all
  converged starts for audit rather than claiming a global optimum.
- Added identity/signed-permutation invariance tests, deterministic tie handling,
  callback-redaction tests, and bounded resource controls.

#### Rotation scoreability assessment

- Added a provider-neutral `assess_rotation_scoreability(...)` gate that combines
  factor-retention rationale, multistart stability, loading-complexity evidence,
  and optional substantive-factor labels into an explicit scoreability decision.
- The gate is deliberately conservative: it does not infer validity or universal
  rotation correctness and routes insufficient or contradictory evidence to human
  review.

#### Formal model-comparison evidence

- Added Rust-backed nested likelihood-ratio tests, AIC, BIC, and corrected AIC;
  relation-safe comparison outcomes distinguish nested, non-nested, and
  observationally equivalent candidates.
- Added deterministic input validation, known-value tests, Rust/Python delegation,
  and equation-to-source traceability; no automatic model preference is emitted
  without distinguishability and optional predictive evidence.

#### Reference-free RAG evaluation measurement

- Added Rust-backed, claim-level reference-free RAG measurement primitives for
  lexical support overlap, signed support/refutation polarity, weighted coverage,
  abstention-aware utility, and exact source-citation coverage.
- Added deterministic known-value, permutation-invariance, malformed-input,
  missing-evidence, and Python/Rust delegation tests.
- Measurements remain descriptive and claim no universal quality threshold,
  semantic entailment, validity, causal utility, or readiness for consequential
  automation.

#### Automated essay evaluation evidence

- Added Rust-backed criterion-level agreement and effect diagnostics for automated
  essay evaluation: exact and adjacent agreement, quadratic-weighted kappa,
  Pearson correlation, standardized mean difference, and subgroup gaps.
- Added deterministic known-value, label-shift, permutation-invariance,
  constant-vector, malformed-input, and Python/Rust delegation tests.
- Correlation is explicitly descriptive and cannot alone establish agreement,
  construct validity, fairness, causal effects, or readiness for consequential
  automated scoring.

### Changed

#### Release cut 0.1.1

- Project version is bumped to 0.1.1 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (bifactor analytics and public API compatibility) now form the
  `[0.1.1] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.1.1] - 2026-08-04

### Added

#### Bifactor analytics

- Added Rust-backed bifactor loading utilities and Schmid-Leiman decomposition
  with deterministic validation, public Python wrappers, and composed PyO3 module
  registration.
- Added complete docstrings plus known-value, invariance, malformed-input,
  import-surface, and Python-to-Rust delegation tests.
- Added APA 7th references and equation-to-source traceability against Schmid and
  Leiman (1957), Jennrich and Bentler (2011), Reise (2012), and Rodriguez et al.
  (2016).
- Retained conservative interpretation: outputs do not establish score validity,
  subgroup fairness, causal effects, or readiness for high-stakes use.

### Changed

#### Release cut 0.1.0

- Project version is bumped to 0.1.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (bifactor analytics and public API compatibility) now form the
  `[0.1.0] - 2026-08-04` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.1.0] - 2026-08-04

### Added

#### Initial release

- Added Rust-first MLSIRM/MLS2PLM simulation, fitting, recovery diagnostics,
  factor-rotation, bifactor, item-scoring, reference-free RAG measurement,
  automated essay-evaluation diagnostics, rubric contracts, CLI workflows, and
  composed PyO3 module registration.
- Added deterministic true-parameter recovery, invariance, malformed-input,
  Rust/Python delegation, GPU parity, property-based, performance, packaging,
  and branch-coverage gates.
- Added APA 7th documentation and conservative interpretation boundaries.

[0.7.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ContextualWisdomLab/fast-mlsirm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ContextualWisdomLab/fast-mlsirm/releases/tag/v0.1.0
