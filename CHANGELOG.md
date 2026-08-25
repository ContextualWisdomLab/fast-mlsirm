# Changelog

## Unreleased

<!-- BEGIN AUTHORITATIVE CHANGELOG FRAGMENTS -->
### Changed

#### Release cut 0.9.0

- Project version is bumped to 0.9.0 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.9.0] - 2026-08-24` release section: new governed contracts (cross-engine
  conformance inventory/provenance/manifest-replay evidence, an external
  validation and transportability profile, a governed structural-model
  pair-decision gate, buyer-facing item-bank lifecycle reports), a new
  Rust-owned crossed/weighted multiple-membership person-effects estimator
  (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001) with CPU-threaded and
  optional GPU kernels, reproducible release-tag-bound PyPI sdist/wheel
  publishing, restriction of production backend selection to Rust-owned
  paths (NumPy parity moved behind an explicit `fit_reference` API), a Rust
  1.97.1 toolchain pin across verification, and a broad continuation of the
  hostile-callback/conversion-protocol hardening sweep across dozens of public
  entry points (CAT, ATA, DIF, equating, scaling, reliability, multilevel,
  response-time, fit-statistics, inference, linking, LLM-judge orchestration,
  parallel-analysis, and rotation/loader concurrency, among others).
- This cut also removes the stale, never-rendered `release-0.8.0-cut.md`
  fragment left over from the abandoned 0.8.0 release attempt (that version
  was never actually tagged or published); its already-recorded
  `[0.8.0] - 2026-08-17` section in `CHANGELOG.md` is left untouched as
  history, and this release supersedes it directly.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
<!-- END AUTHORITATIVE CHANGELOG FRAGMENTS -->
## [0.9.0] - 2026-08-24


### Added

#### Cross-engine conformance inventory contract

- Add a provider-neutral, source-free `ConformanceInventory` contract for independent numerical conformance coverage. The first slice records public estimands, parameterization and identification scope, isolated engine/version/license identity, versioned parameter-mapping and fixture/environment fingerprints, and explicit passed/failed/indeterminate/not-executed states without adding external engines as runtime, build, package, or release dependencies. This is Python validation/provenance schema work only; production psychometric and statistical arithmetic remains Rust-owned.
- Accept both full Git SHA-1 and SHA-256 commit identities so protected-main and harness provenance remains valid across repository hash-format migrations.
- Require at least one executed evidence row before a capability can claim
  `covered` or `partially_covered` status.
- Revalidate exact package-owned engine, evidence, capability, and inventory records before manifest or fingerprint replay so post-construction field rebinding cannot bypass semantic-control, fingerprint, or collection admission; hostile enum controls and container subclasses fail closed before their callbacks execute.

#### Cross-engine conformance provenance

- Add optional run-level conformance provenance for the isolated harness
  commit, environment, RNG seeds, parameter-mapping schema, tolerance
  rationale, output fingerprints, and license classification without storing
  raw responses or adding an external-engine dependency.
- Revalidate exact run-provenance state before direct manifest replay so
  post-construction container rebinding fails closed before caller callbacks.

#### External validation profile contract

- Add a provider-neutral, source-free `ExternalValidationProfile` contract for preregistered external-validity and transportability evidence. The first slice keeps technical, construct, transportability, fairness, and decision-utility evidence distinct; preserves explicit failed/indeterminate/not-executed states; fingerprints normalized manifests; accepts provider-neutral dataset/site identities; and rejects evidence unavailable at the declared analysis cutoff. This is validation/provenance schema work only and does not move psychometric or statistical production arithmetic out of Rust.
- Reject caller-defined profile and evidence-record subclasses before reading their fields, keeping the immutable manifest boundary free of executable attribute callbacks.
- Reject overlapping development, internal-validation, and external-validation dataset identities so a transport claim cannot silently reuse a declared development cohort.
- Revalidate exact profile and evidence state before manifest or fingerprint replay so post-construction field rebinding cannot introduce hostile enum or container callbacks or make a manifest fingerprint disagree with its emitted payload.

#### Cross-engine runtime and redistribution provenance

- Bind cross-engine conformance runs to an explicit container-image or environment-lock identity, operating system, architecture, model-configuration digest, convergence-controls digest, and redistribution status.
- Include the new source-free runtime identities in deterministic manifests and inventory fingerprints while preserving exact-record replay validation against post-construction mutation.
- Keep external-engine evidence isolated from production numerical ownership; no psychometric or statistical arithmetic moves out of Rust.

#### Strict conformance manifest replay

- Added `ConformanceInventory.from_manifest()` and `from_json()` to rehydrate persisted cross-engine conformance evidence through exact package-owned validation.
- Persisted manifests now fail closed on unknown or missing nested keys, caller-defined mapping/list/text subtypes, duplicate JSON object keys, non-finite JSON constants, oversized JSON payloads, fingerprint tampering, and non-canonical normalized content.
- Replay remains provenance and serialization only; production psychometric and statistical arithmetic remains Rust-first.

#### Accessible cross-engine conformance evidence

- Add a deterministic standalone HTML and canonical JSON renderer for strict `ConformanceInventory` manifests, exposing capability coverage, capability × engine execution evidence, immutable inventory/run provenance, limitations, and explicit no-evidence states with exact values in text.
- Add a deterministic provenance-bound long-form JSON table so buyers can download one flat row per capability × engine evidence record without spreadsheet formula execution risk; capabilities with no independent engine remain explicit `not_executed` rows instead of disappearing or turning green.
- Escape manifest text, emit semantic table captions/headers and a restrictive no-script CSP, and state explicitly that numerical conformance is not construct validity, fairness, or high-stakes approval.
- Delegate all ingestion to strict manifest replay and keep the renderer reporting-only; no likelihood, discrepancy, RMSE/MAE, uncertainty, alignment, scoring, or other production psychometric/statistical arithmetic moves out of Rust.

#### Bind Figma buyer evidence to an authoritative ADR

- Record the buyer-review Figma File ID, packet-validation boundary, and
  downstream Code Connect ownership in ADR-0016 and the governance index.

#### Reproducible PyPI release publishing

- Added release-tag-bound sdist and wheel publication with a project-version provenance check, pinned Maturin and PyPA publisher revisions, and persisted checkout credentials disabled.
- The canonical release-tag workflow now explicitly dispatches package publication from the immutable tag, avoiding reliance on release events created with `GITHUB_TOKEN`, which do not recursively start ordinary event-triggered workflows.
- Isolated GitHub release-asset mutation from PyPI credentials, removed the unpinned runtime Twine installation path, and kept duplicate GitHub release assets and PyPI filenames fail-closed rather than silently replacing an immutable release artifact.
- PyPI publication now depends directly on the verified build artifacts rather than successful GitHub asset attachment, so a failed PyPI publication can be retried even when immutable release assets already exist and correctly reject replacement.

#### Crossed multiple-membership person effects

- Added a Rust-owned MAP estimator of crossed / weighted multiple-membership person effects `u_h` (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001). Persons may belong to several groups at once; one-hot nesting remains the singleton special case of the same sparse design.
- Added a CPU-multithreaded Bernoulli score/information reduction and an optional wgpu GPU kernel for that hot loop, with f64 CPU fallback when no adapter is present. Sparse Newton accumulation stays on CPU. This slice does not estimate OLS or AR longitudinal states.
- Added `fast_mlsirm.multilevel.estimate_crossed_person_effects` and `CrossedPersonEffectResult` as marshal-only Python access, plus a true-parameter RMSE recovery test against simulated crossed membership weights.
- Enforced the binary-response contract before native discovery and again inside the Rust estimator: finite non-negative observed cells must be exactly `0` or `1`; negative and non-finite cells retain the established missing-data semantics.

#### Govern structural-model pair decisions

- Add a governed structural-model selection gate that keeps factor retention separate from structure choice, requires explicit parameter-space relation evidence, refuses pairwise selection before the relation-appropriate LR/bootstrap/Vuong procedure, and gates any winner on recovery and intended-score interpretation evidence. The new Python surface performs validation and policy orchestration only; numerical comparison and psychometric arithmetic remain Rust-owned.

#### Add buyer-facing item-bank lifecycle reports

- Added deterministic JSON and standalone accessible HTML reporting for complete governed item-bank lifecycle lineages, including current state, rubric/blueprint provenance, approved-use scope, evidence-class inventory, transition timeline, and explicit missing-evidence limitations.
- Cross-version comparability is reported only as supported when governed linking evidence is present; the report never infers comparability from a nominal score range or active lifecycle state.
- Reporting remains provenance-only: calibration, fit, DIF, information, linking, exposure, drift, and uncertainty arithmetic are referenced by exact evidence identity and are not recomputed in Python.

### Changed

#### Govern non-psychometric item-bank suspension concerns

- Governed item-bank suspension and reactivation can now bind exact non-psychometric concern evidence for evidence/content validity and security/privacy findings, alongside existing DIF, drift, exposure, and linking evidence, without fabricating psychometric drift evidence.
- Suspended records bind the exact newly asserted concern classes into their content-addressed identity, and reactivation requires fresh evidence for those same classes so unrelated evidence cannot clear a quarantine.
- Reactivation rejects a historical approval or concern fingerprint even when it is presented under a replacement evidence identifier; every required reactivation artifact must bind new evidence content.

#### Production backend boundary

- Restrict production `FitConfig` and CLI backend selection to Rust (`rust` or
  fail-closed `auto`). Move the NumPy parity fit behind the explicit
  `fast_mlsirm.fit_reference` API and `fit --reference` mode, preserving
  testable parity without allowing an implicit production owner switch.
- Record the resolved Rust backend for the plain unidimensional MMLE fast path
  so CLI JSON and saved fit summaries report the execution owner rather than
  the unresolved `auto` selector.

#### Harden remaining equating controls before native discovery

- Validate circle-arc method/point/scalar controls, nominal-weights score ceilings and synthetic-population weight, and the composite-linking exponent before compiled-core discovery.
- Reject caller-defined scalar/container subclasses and arbitrary conversion providers without executing their conversion, comparison, representation, hashing, or iteration callbacks.
- Preserve exact built-in and genuine NumPy scalar compatibility while keeping circle-arc geometry, nominal-weights moments, composite-linking weight arithmetic, and all result-affecting equating mathematics in Rust.

#### Dedicated GRM recovery evidence retention

- Kept the 500-replication multidimensional Graded Response Model recovery
  study out of pull-request CI and out of the generic 1,800-second ignored-shard
  budget, then published its printed bias, RMSE, convergence, and theta
  correlation lines as a 90-day Actions artifact.
- Withheld checkout credentials from every Statistical Studies job so
  repository-controlled `cargo test` cannot reuse the Actions token.

#### Pin Rust 1.97.1 across verification

- Pin local Rust builds, Python/Rust package verification, ordinary Rust tests, GPU smoke, packaging, and scheduled statistical studies to exact Rust 1.97.1 instead of a floating stable channel.
- Track the root `rust-toolchain.toml` through Dependabot so future stable compiler updates arrive as reviewable pull requests with exact-head scientific, package, GPU, security, and recovery evidence.
- Preserve the existing public crate compatibility boundary by not adding or raising `package.rust-version`; this is a repository build-baseline change, not a new downstream MSRV claim.

### Fixed

#### Harden bounded subprocess cleanup

- Keep governance and procurement subprocess capture bounded across stdout, stderr, execution time, decoding, and JSON parsing. POSIX cleanup now avoids re-signalling an already reaped process group, successful capture closes parent-side pipe descriptors deterministically, and timeout/overflow paths retain fail-closed evidence without weakening repository gates.

#### Keep judge runtime validation active under Python optimization

- Replace production judge and calibration invariants that relied on removable `assert` statements with explicit package-owned `ValueError` or `RuntimeError` failures, and verify that invalid response-schema admission remains fail-closed under `python -O`.

#### Harden S-X² scalar control admission

- Reject caller-defined integer and floating subclasses at the public S-X² control boundary before numeric conversion or compiled-core dispatch, while preserving exact built-in and concrete NumPy scalar compatibility and leaving all S-X²/G², quadrature, and BH/FDR arithmetic Rust-owned.
- Reject built-in or concrete NumPy integer-valued real controls when float64 normalization would change the integer identity, so `min_expected`, `fdr_q`, and `min_effect` cannot be silently rounded before domain validation.

#### Item-bank transition replay callback safety

- Lifecycle transition replay now validates the exact creation-time record and evidence-reference instance state before invoking canonical serialization or fingerprint verification.
- Frozen lifecycle records mutated through Python object internals cannot shadow `_content_dict()` or evidence `to_dict()` callbacks to execute caller code while acquiring transition authority.
- This changes provenance/integrity validation only; calibration, fit, DIF, item-information, linking, exposure, drift, uncertainty, and other production psychometric arithmetic remain Rust-owned and unchanged.

#### Validate G-theory controls and score evidence before Rust discovery

- G-theory D-study sizes and `Phi(lambda)` scalar controls now fail closed before caller-owned score-array materialization and before compiled Rust capability discovery when invalid, while preserving the existing callback-free Python/NumPy scalar contract.
- D-study control containers now admit exact built-in list/tuple values and exact NumPy signed/unsigned integer arrays of the documented rank before iteration or pair unpacking, so caller-defined sequence, ndarray-subclass, and pair callbacks cannot run while `n_i_prime` / `n_prime` semantics are being established; concrete Python/NumPy integer entries remain supported.
- `gtheory_pi()`, `gtheory_pio()`, and `phi_lambda()` now reject callback-bearing array providers, non-real storage, and complex score evidence before NumPy real narrowing or Rust discovery; ordinary exact NumPy real arrays and built-in list/tuple score trees containing concrete Python/NumPy real scalars remain supported.
- G-study ANOVA/EMS, variance-component, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.

#### Harden Rudner/Lee cut-score control admission

- Validate and materialize Rudner and Lee cut-score scalars before compiled Rust capability discovery, rejecting booleans, caller-defined scalar subclasses, protocol coercion providers, malformed containers, non-finite values, and conversion overflow without invoking caller conversion hooks while preserving exact built-in and concrete NumPy real scalar compatibility. Both public paths now use one canonical package-owned normalizer; cut ordering/domain checks and all classification arithmetic remain Rust-owned.

#### Seal enterprise request record admission

- Enterprise issue scoring-request provenance now rejects caller-defined issue, stakeholder-perspective, and candidate-intervention record subclasses before reading their fingerprints or fields, preventing caller callbacks from executing during canonical record admission while preserving exact package record behavior.

#### Seal enterprise observation admission

- Enterprise issue observation admission now rejects caller-defined scoring-request, evidence-reference, and status-string subclasses before reading provenance or performing enum lookup, preventing caller callbacks during semantic validation while preserving exact package records and serialized status strings.

#### Seal enterprise explicit-value integer admission

- Reject caller-defined integer subclasses for enterprise explicit-value source offsets and deterministic parser record limits before comparison or coercion callbacks can execute, while preserving exact built-in integer domains and stable validation errors.

#### Seal scoring engine-authorization record admission

- Reject caller-defined assessment, scoring-request, and engine-descriptor subclasses before authorization policy or provenance fields are read, preserving exact package records, stable validation errors, and existing engine-policy semantics.

#### Seal assessment aggregate record admission

- Assessment assembly now rejects `ConstructSpec`, `RubricSpecification`, and scoring-policy subclasses before reading package-owned provenance or construct-scope fields, preventing caller-defined attribute/fingerprint callbacks from executing during aggregate contract admission while preserving exact package records and existing cross-reference semantics.

#### Bifactor scoreability control trust boundary

- Hardened both public bifactor scoreability entry points so `general_factor` and `zero_tolerance` are validated and normalized before loading, uniqueness, or logit-slope materialization and before compiled-core discovery.
- Reject booleans, caller-defined numeric subclasses, and arbitrary conversion-protocol objects without executing their callbacks, while preserving concrete Python/NumPy scalar compatibility and Rust ownership of index/domain validation and all scoreability arithmetic.

#### Scoring shared enum callback safety

- Shared scoring enum admission now preserves exact enum members and accepts only exact built-in strings for serialized enum values before invoking Enum lookup.
- Caller-defined string subclasses and arbitrary non-text objects fail closed with the existing package-owned assessment error before hostile hash or equality callbacks can run.
- Added public EngineDescriptor regressions proving callback-free rejection while preserving built-in string and exact enum-member compatibility; no scoring, calibration, likelihood, estimator, ranking, utility, or psychometric arithmetic changed.

#### Model-spec record admission

- Model resolution now admits only exact package-owned exploratory and confirmatory model records before reading their fields, so caller-defined model-spec subclasses cannot execute attribute callbacks during validation. Exact built-in/concrete NumPy factor counts and exact package model records retain their existing behavior; multidimensional exploratory estimation remains separately governed by #633.

#### Correct skewed-population Mokken study contract

- Keep the normal-trait Monte Carlo condition as the calibrated H/recovery
  contract.
- Standardize the positive-skew half-normal latent condition to the same
  location and scale as the normal condition before applying the shared 1.5
  theta scale, so the study changes distribution shape without confounding
  skewness with the previous approximately 28% narrower latent spread.
- Require both moment-matched latent conditions to retain the calibrated
  Loevinger H band, while keeping AISP full-recovery acceptance calibrated on
  the normal condition rather than treating the user-selected `c = 0.3`
  cutoff as distribution-invariant.
- Preserve the exact ignored-study execution and report failures normally.
- Declare that the workflow consumes no secrets and require reviewed
  `${{ secrets.NAME }}` environment injection for any future credentialed
  study.

#### Seal bounded JSON semantic-input callback boundaries

- Reject caller-defined byte/depth limit integers before comparison and caller-defined JSON text subclasses before encoding, while preserving exact built-in controls, bounded parsing semantics, and the existing descriptor/path/size/depth defenses used by repository release and governance automation.

#### Factor-retention callback safety

- Hardened governed factor-retention evidence admission so caller-defined integer and evidence-record subclasses are rejected before comparison or record-field callbacks can execute, while preserving built-in candidate counts and existing conservative retention semantics.

#### Harden multilevel text callback safety

- Require exact built-in strings for contextual schema versions, descriptive identifiers, and provenance fingerprints before comparison, normalization, regex, or encoding work, preventing caller-defined `str` subclasses from executing callbacks during multilevel and temporal contract admission.

#### Make repository test imports deterministic

- Pytest now exposes both the repository root and the Python source tree from
  committed configuration, so tests that materialize repository automation
  scripts do not require an operator-specific `PYTHONPATH=.` workaround.
- Agent guidance now derives its advertised Python support floor from the same
  `pyproject.toml` requirement guarded by repository tests, preventing stale
  lower-version setup instructions from diverging from package metadata.

#### Executed conformance provenance integrity

- Fail closed when a cross-engine conformance inventory contains executed `passed`, `failed`, or `indeterminate` evidence without exact run provenance.
- Require both raw-output and normalized-output SHA-256 identities for executed conformance runs while preserving optional output hashes for genuinely nonexecuted plans.
- Revalidate nested run provenance before applying the execution consistency gate so post-construction mutation cannot bypass package-owned admission.

#### Fail closed on missing release source identity

- The buyer-facing release evidence index now rejects timed-out, failed, unavailable, empty, malformed, or non-canonical Git `HEAD` identity instead of allowing an otherwise complete packet to report `status: "ok"` with unreconstructable source provenance.
- Valid repositories continue to record the exact full lowercase hexadecimal source commit without changing psychometric/statistical numerical ownership.

#### Require reconstructable buyer-packet source identity

- Buyer evidence packet generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated/malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving interoperability without changing psychometric/statistical numerical ownership.

#### Require reconstructable benchmark source identity

- Benchmark evidence generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated/malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving repository interoperability without changing psychometric/statistical numerical ownership.

#### Figma evidence source provenance

- Figma design-evidence manifests now fail closed when the repository source commit cannot be resolved to a canonical full lowercase SHA-1 or SHA-256 object identity, instead of emitting buyer-facing evidence with `source_commit: "unknown"` or an abbreviated/malformed revision.

#### Workflow registry transport failures

- The read-only workflow-registry audit now converts missing or inaccessible local GitHub CLI execution into a stable fail-closed `GitHubApiError`, so automation can emit bounded failure evidence instead of crashing with raw operating-system details.

#### Commercial release source identity

- Fail commercial release evidence generation closed when the source Git revision is unavailable or malformed, and require a canonical lowercase full SHA-1 or SHA-256 identity before provenance can be emitted.

#### Enterprise gate source-provenance hardening

- Require enterprise due-diligence manifests to bind `source_commit` to a canonical lowercase full SHA-1 or SHA-256 Git object identity instead of accepting abbreviated or arbitrary printable identifiers.
- Reject caller-defined string subclasses before text callbacks can execute at the source-provenance admission boundary, so a successful gate remains reconstructable from exact source identity.
- Restrict manifest output to a relative path inside the invocation directory and reject symlinked or tree-escaping destinations before writing.
- Write through a validated descriptor tree into a same-directory temporary file and atomically rename it into place on supported POSIX systems, so a failed write cannot truncate the previously accepted manifest.
- Preserve an existing manifest's access permissions across atomic replacement and use ordinary process file-creation permissions for a new manifest instead of forcing buyer-facing evidence to owner-only mode.

#### Enterprise gate semantic-control callback safety

- Reject caller-defined string subclasses for enterprise gate names and currency codes before normalization can invoke caller text callbacks.
- Reject caller-defined integer subclasses for procurement scenario amounts before comparison while preserving the positive-integer validation contract for exact built-in values.

#### Changelog fragment marker integrity

- Reject authoritative changelog fragments containing reserved managed-block marker literals before rendering or update, preventing nested markers from producing a changelog that fails its own next integrity check.

#### Scoring fingerprint text admission

- Require caller-supplied SHA-256 scoring provenance to be an exact built-in string before validation or retention, preventing valid-looking string subclasses from crossing the package trust boundary as canonical fingerprints.
- Apply the same exact built-in text boundary to structured scoring error code, path, and message fields.
- Reject caller-defined scalar subclasses in bounded scoring metadata before canonicalization or digesting.

#### Reject ambiguous duplicate JSON artifact members

- The shared bounded artifact JSON loader now rejects duplicate object member
  names at every nesting level instead of accepting last-value-wins semantics,
  while preserving its existing stable-file, UTF-8, byte, nesting, and parser
  controls.

#### Strict artifact JSON constants

- Reject `NaN`, `Infinity`, and `-Infinity` by default in the shared bounded artifact JSON loader so persisted package artifacts use interoperable JSON semantics; explicit caller `parse_constant` policies remain supported.

#### Require interoperable bounded JSON artifacts

- Reject duplicate object member names and non-standard non-finite numeric constants in the shared repository-automation bounded JSON reader, so file-backed and direct parsing use the same unambiguous RFC-compatible semantics while preserving existing size, depth, UTF-8, path-identity, and callback-safety controls.

#### Seal bounded subprocess command admission

- Reject caller-defined command-container and text-token subclasses before repository automation materializes or checks command arguments, preventing validation-time callback execution while preserving exact built-in list and tuple vectors.

#### Population-label narrowing safety

- Reject multigroup and multilevel population labels that cannot round-trip through signed 64-bit integer representation before compaction, preventing narrowing overflow from silently reordering the identified reference population while preserving valid sparse labels and the signed `int64` boundary.

#### BRATT control admission

- Validate and normalize Bradley-Terry-with-ties reference, iteration, and tolerance controls before comparison-data materialization or compiled-core discovery, rejecting callback-bearing scalar subclasses and protocol providers while preserving trusted built-in and NumPy scalar inputs.
- Keep BRATT probability, MM-update, reference-rescaling, convergence, and log-likelihood arithmetic unchanged in the Rust core.

#### RAG evidence limitation replay integrity

- Replay factory-derived RAG evidence limitation records before manifest or fingerprint projection so post-construction mutation fails closed before caller callbacks can execute.

#### Response-time calibration semantic control safety

- Reject caller-defined numeric and truth-value protocols before response-time calibration controls are normalized or dispatched to the Rust core.
- Require the joint speed-accuracy Gauss-Hermite node count to be an exact supported integer instead of silently narrowing floating-point values.
- Keep required positive-finite runtime validation active under optimized Python execution instead of relying on `assert` guards that disappear with `-O`.
- Preserve positive-finite stopping, variance-floor, and fixed-speed-scale contracts while keeping all response-time likelihood and estimation arithmetic Rust-owned.

#### Polytomous fit semantic control safety

- Reject caller-defined text, integer, real, and hashing protocols before GRM/GPCM calibration controls are normalized, response data are materialized, or the Rust core is discovered.
- Require calibration quadrature to use an exact supported integer node count rather than callback-capable membership or lossy coercion.
- Normalize both `NaN` and `-1` as missing polytomous responses before category validation, and report malformed response conversion through a stable package-owned numeric-input error.
- Preserve the category, iteration, and positive-finite stopping contracts while keeping the Bock-Aitkin EM/Newton estimator and all result-affecting psychometric arithmetic Rust-owned.

#### Keep essay-report pointer focus modality-safe

- Suppress pointer-acquired outlines on focusable essay-report table regions and canonical JSON blocks only when `:focus-visible` is false. Keyboard navigation retains the explicit high-contrast focus indicator, and regressions reject blanket `:focus { outline: none; }` suppression.

#### Reject overflowing polytomous DIF labels

- Polytomous DIF group and studied-item label/index vectors now verify signed-64-bit narrowing before compaction or Rust dispatch, preventing unsigned boundary values from wrapping negative and changing group/reference identity.
- Valid non-negative signed-64-bit and sparse/non-contiguous labels remain supported; GRM/GPCM DIF likelihood and statistical arithmetic remain Rust-owned and unchanged.

#### CAT administration data integrity

- Reject administered item indices that cannot be represented losslessly as signed 64-bit identities before range/mask handling, and reject complex-valued binary responses before any real-valued coercion can discard their imaginary component. Ordinary signed indices and real 0/1 responses retain the existing Rust-owned CAT likelihood, ability-estimation, and information paths.

#### Complex-valued polytomous response admission

- Reject complex-valued polytomous response matrices before any `float64` narrowing can discard imaginary components and turn a different observed category into a valid-looking real category.
- Preserve real integer categories plus `NaN` and `-1` missingness semantics across calibration, scoring, DIF, item/person fit, and other callers of the shared response-admission boundary without changing Rust-owned psychometric arithmetic.

#### CRM response data integrity

- Reject complex-valued continuous-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component, and reject object-dtype response storage before caller-defined numeric conversion can run.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees with package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Exact numeric NumPy arrays nested as inert rows inside built-in containers remain compatible without admitting ndarray subclasses or object/text leaves.
- Preserve `NaN` as the CRM missing-cell marker while rejecting `+Infinity` and `-Infinity` before native discovery instead of silently reclassifying those invalid observed values as missing. Ordinary finite real-valued evidence retains the existing Rust-owned CRM fitting path.
- Bound CRM response evidence to 20,000,000 logical cells before sequence materialization or dense real-valued work. Exact broadcast arrays and exact NumPy row leaves nested in trusted built-in matrices are rejected from shape/size metadata before allocation; shared acyclic built-in subtrees retain logical-occurrence accounting without exponential re-traversal.

#### IRTree scientific-evidence admission

- Reject complex-valued IRTree response matrices, tree mappings, and node-dimension vectors before any `float64` narrowing can discard imaginary components and change observed categories, mapping branches, or factor assignments.
- Reject arbitrary NumPy array providers, callback-bearing container/scalar subclasses, and object/text storage before package-triggered `__array__` or numeric-conversion callbacks can synthesize or replace IRTree evidence.
- Preserve exact NumPy real-numeric arrays plus exact built-in list/tuple evidence containing package-trusted Python/NumPy real scalars, including ordinary `NaN` missingness, without changing IRTree mapping semantics or psychometric estimator arithmetic.

#### Complex-valued curvature admission

- Reject complex-valued Hessian and covariance matrices before any `float64` narrowing can discard imaginary components and alter second-order, covariance, or standard-error evidence.
- Keep eigendecomposition, inversion/pseudoinversion, and standard-error arithmetic in the Rust core while preserving existing real square-matrix contracts.

#### Oakes uncertainty input admission

- Reject complex-valued response matrices and factor assignments before any real/integer narrowing can discard imaginary components in the public Oakes standard-error wrapper.
- Preserve existing binary-response missingness and integer factor semantics while keeping Oakes information, finite-difference, inversion, and standard-error arithmetic in the Rust core.

#### Oakes factor-id signed-64 admission

- Reject Oakes `factor_id` values that cannot round-trip through signed 64-bit integer marshalling before dimension derivation or Rust uncertainty arithmetic, preventing unsigned overflow from silently changing item-to-dimension assignments.

#### WLE complex-evidence admission

- Reject complex-valued dichotomous and polytomous WLE responses and item parameters before real-valued marshalling or Rust scoring dispatch, preventing imaginary components from being silently discarded.

#### Seal LLTM data and control admission

- Reject complex-valued LLTM response matrices and explanatory-design weights before real-valued narrowing can discard their imaginary components.
- Validate Boolean, iteration, and tolerance controls before caller-owned data materialization or compiled-Rust capability discovery, while preserving trusted built-in and concrete NumPy scalar inputs and the Rust-owned LLTM estimator.

#### Nominal-response admission hardening

- Validate nominal category, quadrature, iteration, tolerance, Monte Carlo point, and RNG-seed controls before caller response materialization, accepting only package-trusted built-in or concrete NumPy scalar identities and passing normalized primitives to Rust.
- Reject complex response evidence before real-valued narrowing and reject infinite response values instead of silently reclassifying them as missing, while preserving ordinary real/integer categories plus documented NaN/negative missingness.
- Keep nominal probabilities, marginal likelihood, estimation, integration, convergence, identification, and EAP arithmetic unchanged in the Rust numerical core.

#### GPCM admission hardening

- Validate GPCM category, quadrature, iteration, tolerance, integration-point, and RNG-seed controls before caller response materialization, admitting only package-trusted built-in or concrete NumPy scalar identities and passing normalized primitives to Rust.
- Reject complex response evidence before real-valued narrowing and reject infinite response values instead of silently reclassifying them as missing, while preserving ordinary categories plus documented NaN/negative missingness.
- Keep GPCM probabilities, marginal likelihood, estimation, integration, reflection/identification, convergence, and EAP arithmetic unchanged in the Rust numerical core.

#### Mixture-response admission hardening

- Reject complex mixture-IRT response evidence before real-valued narrowing so caller data cannot silently project onto a different observed 0/1 pattern before Rust validation.
- Reject object-dtype response storage before per-element numeric coercion, including Python complex objects and caller-defined conversion callbacks, with the package-owned real-valued input error.
- Reject positive and negative infinity instead of treating them as undocumented missing responses, while preserving `NaN` as the documented MAR missingness representation.
- Keep mixture likelihood, posterior, EM updates, restart selection, canonical class ordering, convergence, and EAP arithmetic unchanged in the Rust numerical core.

#### KSIRT input admission

- Validate and normalize KSIRT kernel/grid controls before caller array materialization or compiled-core discovery, reject complex response or bandwidth evidence before real-valued `float64` marshalling, reject object/string-like storage before per-element numeric conversion can execute caller callbacks, and reject arbitrary array-protocol providers before NumPy materialization while preserving exact NumPy arrays and plain built-in numeric sequences. The Nadaraya-Watson/OCC estimator and all production psychometric/statistical arithmetic remain Rust-owned.

#### Mixed-format response admission

- Reject complex-valued mixed-format response evidence before real-valued marshalling so imaginary components cannot be silently discarded before categorical validation and Rust-owned calibration.

#### Subscore complex-evidence admission

- Reject complex-valued response and subscale-assignment evidence before real-valued marshalling so imaginary components cannot be silently discarded before Rust-owned Haberman subscore analysis.

#### DETECT evidence admission hardening

- Reject complex or non-real-numeric DETECT response storage before real-valued marshalling so observed binary evidence cannot be silently projected onto different data.
- Reject complex or non-real-numeric DETECT cluster storage before partition normalization so item-to-dimension labels cannot be silently projected onto a different real partition.
- Reject arbitrary response/cluster array-protocol providers before NumPy materialization, while preserving exact NumPy arrays and plain built-in sequences of trusted real scalar values.
- Reject a self-referential or otherwise cyclic list/tuple response or cluster (for example `a = []; a.append(a)`) before flattening instead of looping until the process is killed; cycle detection tracks only the active ancestor path, so legitimate repeated/shared acyclic rows remain accepted.
- Bound compressed shared-DAG list/tuple expansion and exact NumPy-array evidence before further package materialization, preventing hidden expansion or arrays above 20,000,000 logical cells while retaining ordinary shared-row compatibility.
- Preserve Rust ownership of conditional-covariance and DETECT index arithmetic; the Python change is limited to validation and marshalling.

#### Graded-response evidence admission hardening

- Normalize GRM integration, iteration, category, seed, and tolerance controls before caller response materialization, without invoking arbitrary scalar coercion callbacks.
- Reject complex, non-real-numeric, and infinite response storage before real-valued marshalling so observed graded-category evidence cannot be silently projected or reclassified as missing.
- Preserve the documented `NaN`/negative missingness convention, confirmatory loading validation, and Rust ownership of GRM likelihood, integration, parameter estimation, EAP, identification, and convergence arithmetic.

#### Linking evidence admission

- Reject complex-valued or non-real-numeric fixed-item and common-item linking evidence before lossy real marshalling, caller element conversion, or compiled Rust-core discovery; reject non-finite source-theta evidence before fixed-item Rust dispatch while preserving Rust-owned linking arithmetic.

#### Factor input admission hardening

- Reject complex and non-real-numeric factor-analysis, reliability, and Velicer MAP evidence before real-valued marshalling can alter caller data or execute object-element conversion.
- Normalize trusted `n_factors` and `max_m` integer controls before caller array materialization and Rust-core discovery while preserving concrete NumPy integer compatibility.

#### Parallel-analysis data admission

- Reject complex and non-real-numeric caller matrices before Horn/Glorfeld parallel-analysis input is narrowed to `float64`, preventing imaginary evidence from being silently discarded or object-element numeric callbacks from running during package-owned admission.
- Preserve existing real numeric input compatibility, integer-control validation, bounded random-eigenvalue workspace policy, and Rust ownership of eigenvalue, random-benchmark, centile, and retention arithmetic.

#### Validate Hofstee controls before score materialization

- Validate and order the four Hofstee percentage controls before caller-owned score arrays are materialized, so rejected semantic controls cannot trigger score-side array protocols before the package emits its stable validation error.
- Preserve the existing Rust-owned Hofstee ogive, intersection, fallback, and cut-score arithmetic.

#### CAT exposure item-evidence admission

- Reject complex-valued and non-real-numeric Sympson-Hetter and a-stratified item-parameter storage before lossy `float64` marshalling or compiled-core discovery, while preserving ordinary real item banks and Rust-owned CAT exposure algorithms.

#### Seal Chang-Ying KL evidence admission

- Reject complex or non-real-numeric KL item-parameter storage before any lossy `float64` narrowing or Rust-core discovery.
- Require `kl_select()` administration masks to use Boolean storage rather than truth-value coercion.
- Normalize `theta0`, `delta`, and `r` only from package-trusted built-in or concrete NumPy real scalar identities before caller array work.
- Preserve contiguous `float64`/Boolean native marshalling after admission while leaving Chang-Ying KL integration and selection arithmetic Rust-owned.

#### Delta-plot group evidence admission

- Reject non-real-numeric Delta-plot group storage before real-valued coercion, preventing textual reference/focal labels from being silently reinterpreted and object-dtype cells from executing caller numeric callbacks during Python-to-Rust admission.
- Preserve ordinary numeric and Boolean 0/1 group arrays while keeping Angoff Delta-plot psychometric arithmetic unchanged in the Rust core.

#### Owen CAT evidence admission

- Establish Owen posterior/CAT scalar, Boolean, item-array, and binary-response trust boundaries before compiled-core discovery or caller-controlled coercion. Caller-defined scalar/truth callbacks, complex/text/object item or response storage, and arbitrary array providers now fail closed while supported NumPy scalar/array evidence is normalized to inert built-in/contiguous representations. Owen posterior moments, b-matching, variance stopping, and all result-affecting psychometric arithmetic remain Rust-owned.

#### Seal EPV trust-boundary admission

- Reject caller-defined posterior scalar callbacks, lossy or non-numeric EPV item evidence, and non-Boolean administered masks before native dispatch while preserving ordinary NumPy inputs and Rust-owned predictive/variance/selection arithmetic.

#### Seal Sympson-Hetter scalar control admission

- Validate package-trusted `r_max` and `tol` scalar identity and semantic domains before caller item arrays or native discovery, preserving Rust-owned Sympson-Hetter calibration, simulation, update, and stopping arithmetic.
- Preserve the Rust finite `tol >= 0` contract directly in the canonical `exposure.sympson_hetter` boundary and remove the duplicate zero-tolerance marshalling/dispatch shim.

#### Seal SPRT evidence and control admission

- Validate package-trusted Wald SPRT scalar controls and reject coercive, textual, object, or complex item/response evidence before native dispatch, preserving Rust-owned boundaries, likelihood-ratio accumulation, first-crossing decisions, and trace arithmetic.

#### Seal CI-classification evidence and control admission

- Validate package-trusted confidence-interval classification controls and reject coercive, textual, object, or complex item/response evidence before native dispatch, preserving Rust-owned EAP, posterior-SE, interval, and strict first-crossing arithmetic.

#### Flexilevel evidence admission

- Validate Lord flexilevel item-count and platform-size controls before caller response materialization, and reject complex, textual, object-backed, lossy, or domain-invalid response/probability evidence before native-core discovery while preserving supported binary NumPy arrays, plain callback-safe 1-D/2-D list/tuple response array-likes, and finite odd-length probability vectors. Routing, red/blue self-scoring, forward recursion, score-lattice probabilities, mean, and variance remain Rust-owned.
- Preserve callback-safe list/tuple probability compatibility for package-trusted concrete NumPy real scalars as well as built-in real scalars.

#### Observed-score equating evidence admission

- Reject complex, object-backed, and textual score/frequency evidence before lossy `float64` marshalling or compiled-Rust discovery across equivalent-groups, NEAT, kernel, presmoothing, and SEE entry points.
- Preserve real Boolean/integer/unsigned/float evidence while keeping equating, smoothing, uncertainty, and population-linking arithmetic Rust-owned.

#### Fixed-form test assembly admission safety

- Harden fixed-form assembly so form length and content-constraint controls are normalized before caller item evidence, complex/object information cannot be projected through `float64`, content labels are admitted as text without caller stringification, and exclusion indices must fit signed 64-bit item identity without narrowing overflow before the Rust-owned greedy assembly runs.

#### Harden constrained-CAT evidence admission

- Validate CCAT ability, item, content-group, target, and administered-mask evidence before native dispatch; reject callback-bearing or lossy storage, require lossless non-negative integral `uintp` group marshalling, and leave constrained-CAT selection arithmetic Rust-owned.

#### Bound the judge's weighted-score boundary

- `ContextualOrchestratorJudge.judge()`'s plain scoring path (no `category_count`, the simplest public interface) trusted the model's own self-reported top-level `score` for the accept/reject decision instead of deriving it from `criterion_scores` and each `JudgeCriterion.weight`, unlike the three `category_count`-based paths, which already discard the self-reported score in favor of a mechanically recomputed weight-aware average. A model could report a high aggregate score while giving a low score on a heavily-weighted criterion and still be accepted. Made the plain path derive `score` the same way as the other three (issue #1238).
- Rejected a non-finite aggregate criterion weight before any contextual-orchestrator transport call. `JudgeCriterion` validates each weight as finite and positive, but two individually valid weights (for example `1e308` each) could still overflow their sum to infinity; a weighted score could then silently collapse to an incorrect finite value (for example `0.0`) instead of failing closed. All three weighted-score paths now share one bounded, finite denominator (issue #1235).

#### Response-time evidence admission

- Reject complex, object/text, callback-bearing, and arbitrary array-provider response-time evidence before real-valued marshalling or Rust-core discovery across standalone RT calibration, joint speed-accuracy calibration, and RT person-fit diagnostics, while preserving ordinary built-in real-numeric sequence and NumPy-array inputs.
- Replaced the recursive built-in-sequence walk with an explicit stack so a deeply nested response-time list/tuple (past Python's recursion limit) or a self-referential one (`a = []; a.append(a)`) rejects with a validation error instead of crashing the process with an uncaught `RecursionError` or looping forever.

#### Response-time person-fit control safety

- Validate `alpha_level` and `z_fast` with callback-free concrete real-scalar admission and the Rust-owned `(0, 1)` / finite non-negative domains before native-core discovery in response-time person-fit diagnostics.

#### Empirical Bayes DIF evidence admission

- Reject arbitrary array-protocol providers and callback-bearing sequence elements before Empirical Bayes Mantel-Haenszel DIF evidence is narrowed or dispatched, while preserving exact NumPy real-numeric arrays and ordinary built-in real-numeric list/tuple vectors.

#### Nonparametric person-fit response admission

- Reject arbitrary array-protocol providers and callback-bearing response cells before complete dichotomous person-fit evidence is materialized or dispatched, while preserving exact NumPy real-numeric arrays and ordinary built-in real-numeric list/tuple matrices.

#### DIMTEST evidence admission hardening

- Reject arbitrary response and AT1/AT2 array-protocol providers before NumPy materialization so caller callbacks cannot synthesize scientific evidence or subtest membership.
- Preserve exact NumPy real-numeric arrays and plain built-in sequences of trusted real scalars, plus existing complete dichotomous response and integer index semantics.
- Preserve Rust ownership of Stout DIMTEST conditional-variance, bias-correction, p-value, and retained-group arithmetic.

#### Seal paired rating-range evidence admission

- Reject callback-bearing or subclassed caller rating containers before NumPy conversion or Rust-core discovery, while preserving exact NumPy numeric arrays and the existing ordinal category/domain checks. Paired rating-range descriptive arithmetic remains Rust-owned.

#### Reliability evidence admission

- Reject callback-bearing, complex, or non-real-numeric caller evidence before Rust discovery in Guttman lambda, ten Berge mu, Cronbach alpha, and person-separation reliability entry points, while preserving ordinary NumPy arrays and trusted built-in sequence inputs.
- Reject over-nested or cyclic built-in sequence evidence at the public API's known 1-D/2-D rank boundary before NumPy materialization or native discovery, while preserving shared acyclic rows and trusted real-scalar sequence compatibility.
- Use one callback-free masked-array diagnostic across ICC, Guttman lambda, ten Berge mu, Cronbach alpha, person separation, and pairwise-rater reliability so masked evidence consistently tells callers to encode missingness with NaN before any native dispatch.
- Preserve historical built-in sequence compatibility when rows are exact real-numeric NumPy arrays, while retaining callback-free rejection of ndarray subclasses and non-real row storage before materialization.
- Preserve historical rater-sequence Boolean semantics without reopening caller protocols: pure Boolean built-in sequences keep the Boolean-specific diagnostic, while mixed Boolean+numeric built-in sequences retain NumPy's numeric promotion.
- Make reliability-adapter installation recover every primary sibling after an interrupted partial bind instead of treating a hardened ICC wrapper alone as proof that the whole public reliability surface was installed.
- Bound primary and rater reliability evidence to 20,000,000 logical cells before NumPy materialization or contiguous `float64` allocation, including exact broadcast views and exact NumPy leaves nested inside trusted built-in sequences.

#### Pairwise reliability evidence admission

- Validate the Pearson/Spearman pairwise-rater Fisher control and caller-owned ratings evidence before native discovery, rejecting callback-bearing or non-real evidence without changing Rust-owned correlation, ranking, Fisher-transform, or inference arithmetic.

#### ICC ratings evidence admission

- Preserve callback-free ICC semantic controls while also rejecting callback-bearing, complex, Boolean, or non-real ratings before native discovery; trusted numeric arrays and built-in numeric sequences still marshal to the unchanged Rust ICC implementation.
- Preserve the established Boolean-rating diagnostic for trusted built-in/NumPy-Boolean sequences, including mixed Boolean-plus-numeric sequences whose Boolean identity NumPy would otherwise erase by numeric promotion, and preserve actionable `NaN` missingness guidance for NumPy `MaskedArray` subclasses without reopening caller-defined array or scalar callbacks.

#### Remaining reliability rater-evidence admission

- Validate Krippendorff alpha, Finn reliability, Maxwell RE, and Robinson A semantic controls and rater evidence through callback-free package admission before Rust discovery, while preserving trusted numeric sequence compatibility and the existing Rust-owned agreement/reliability arithmetic.

#### Answer-copying evidence admission

- Reject callback-bearing NumPy array providers, ndarray/container subclasses, and caller-defined numeric subclasses before answer-copying evidence is materialized for Wollack omega, K-index/K1/K2/S1/S2, or GBT.
- Preserve exact NumPy numeric arrays and exact built-in list/tuple evidence containing package-trusted Python/NumPy real scalars, while keeping existing complex, dimensional, finite, index, binary, probability, and relation validation contracts.
- Keep all result-affecting answer-copying statistics and tail/regression arithmetic in the Rust numerical core; this change only hardens Python validation and marshalling.

#### Bound G-theory score evidence before dense materialization

- `gtheory_pi()` and `phi_lambda()` now reject score evidence outside the documented two-dimensional persons-by-items shape before dense NumPy materialization; `gtheory_pio()` applies the same fail-first contract to its three-dimensional persons-by-items-by-occasions shape.
- G-theory score evidence now has an explicit 20,000,000-cell logical-resource ceiling that applies to exact NumPy views and trusted built-in sequence trees before a contiguous `float64` copy is allocated.
- Built-in score-tree preflight now advances one child at a time, so transient traversal state is bounded by nesting depth instead of eagerly scheduling every sibling before the logical-cell ceiling can fire.
- Existing exact NumPy arrays, ordinary built-in list/tuple score trees, exact NumPy-array rows, callback-free cycle rejection, and Rust-owned G-study/D-study/`Phi(lambda)` arithmetic remain unchanged.

#### Bound G-theory D-study result-row requests

- `gtheory_pi()`, `gtheory_pio()`, and `phi_lambda()` now reject D-study request vectors above 10,000 rows before score materialization or compiled-core discovery.
- D-study result-row count is bounded independently from the existing 1,000,000 per-prime magnitude ceiling, so small valid prime values cannot be repeated to request an unbounded native result table.
- Exact built-in list/tuple controls, trusted Python/NumPy integer entries, the existing per-prime size bound, and all Rust-owned G-study/D-study/`Phi(lambda)` arithmetic remain unchanged.

#### G-theory NumPy D-study control compatibility

- Preserve exact NumPy signed/unsigned integer arrays for one-facet and two-facet D-study size controls, and preserve exact built-in `range` values on the one-facet `Sequence[int]` surface, while continuing to reject ndarray subclasses, arbitrary array providers, callback-bearing sequence subclasses, Boolean/float/object/text control arrays, malformed rank/shape, non-positive values, and existing resource-limit violations before Rust dispatch.
- Normalize accepted NumPy control arrays and built-in range controls to package-owned built-in integer payloads; G-study, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.

#### Rater reliability installer recovery

- Recover interrupted Krippendorff/Finn/Maxwell/Robinson reliability-adapter installation by requiring the complete package-owned rater wrapper set before idempotent short-circuiting, while preserving callback-free evidence admission and Rust-owned reliability arithmetic.

#### Close CI contract drift on the toolchain pin and metadata scalar admission

- Pin the `grm-recovery` scheduled statistical-study job's `dtolnay/rust-toolchain` step to exact Rust `1.97.1`, closing a gap where it silently floated to the default stable channel while every sibling verification lane stayed pinned.
- Align `test_metadata_normalizes_string_subclasses_without_callbacks` (formerly `test_metadata_rejects_string_subclasses_before_callbacks`) with the metadata scalar admission boundary's actual, intentional behavior: caller-defined `str` subclasses are safely normalized through the inert `str.__str__` descriptor (matching the established `int`/`float` subclass handling in the same function) without invoking any subclass-defined method, rather than being rejected outright.

#### Fail closed on unsafe multilevel contextual effects

- Multilevel contextual-effect evaluation now fails closed when any referenced context random-effect value is NaN or infinite and when finite inputs overflow the weighted sum, preventing non-finite predictor results from escaping the Rust boundary while leaving unreferenced table capacity outside sparse validation work.
- Python context-effect marshalling snapshots each required mapping value once without caller-defined membership probes and normalizes hostile lookup callbacks to non-reflective package errors before native dispatch.

#### Seal governed RAG request replay

- Reject caller-defined `ScoringRequest` subclasses at governed RAG perturbation and facets-calibration replay boundaries before any request field can execute caller code. Exact factory-sealed requests retain the existing provenance validation, while invalid subclasses now fail through stable non-reflective package errors.

#### Harden model-comparison casewise numeric trust boundary

- Harden public non-nested model-comparison casewise value admission so arbitrary float-protocol objects and caller-defined numeric subclasses fail closed without executing conversion callbacks, while preserving exact Python and supported NumPy real scalars; Vuong statistics remain Rust-owned.

#### Multilevel M2 moment and covariance ownership

- Move multigroup and multilevel M2 population-moment integration into the
  Rust/PyO3 numerical boundary, including the shared cluster-intercept
  reduction.
- Move the finite-cluster moment-covariance construction into Rust while
  preserving compact-label validation, finite-cluster correction, and the
  existing M2/RMSEA2 estimand.
- Keep the NumPy implementations available only as explicit parity references;
  public M2 paths fail closed when the required native entry point is absent.

#### Multilevel M2 Rust projection

- Multilevel M2 now routes both fitted-model and cluster-robust independence projections through the compiled Rust core, failing closed when that projection entrypoint is unavailable.

#### Structured M2 Rust ownership

- Route public single-population `m2()` calls that include estimated population
  moments, anchored items, or a fixed spatial coefficient through the Rust/PyO3
  M2 kernel. Missing structured native capability now fails closed instead of
  entering the NumPy reference implementation.
- Preserve the existing M2 estimand and degrees-of-freedom contract while
  moving finite-difference calibration and population nuisance columns into
  the Rust numerical owner.

#### Workflow-registry audit transport retry hardening

- Expanded the read-only Actions-registry audit transport's bounded retry classifier to cover transient HTTP 403, 404, 429, and all 5xx responses, while preserving fail-closed exhaustion and immediate failure for non-transient authentication errors such as HTTP 401.
- Added direct transport regression coverage so incident audits do not misclassify one transient GitHub control-plane response as a completed inventory failure.

#### Harden RAG metadata callback safety

- Validate caller-provided RAG metadata keys exactly once before reading any values, then freeze only the captured allowlisted values. Hostile membership, key/value, duplicate-key, and key-reiteration callbacks now fail through non-reflective package errors without granting new metadata authority.

#### Exposure-control scalar callback safety

- Validate CAT/exposure integer controls from exact built-in Python and genuine NumPy scalar types before caller-dispatchable coercion or Rust-core discovery, preserving integral built-in/NumPy floating controls, package-owned bounds/errors, and Rust-owned exposure, routing, scoring, posterior, recovery, and simulation arithmetic.

#### Harden scoring-policy integer callback boundaries

- Reject caller-defined integer coercion at scoring-policy positive-integer boundaries before any `__index__` callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError` semantics.

#### ATA integer callback safety

- Automated test assembly now admits only exact built-in integers and explicitly supported genuine NumPy integer scalar identities for public length, seed, exposure, content-count, and exclusion controls before normalization.
- Caller-defined Python and NumPy integer subclasses fail closed before conversion callbacks or item-information work, while existing finite-domain validation and genuine NumPy scalar compatibility are preserved.
- Added focused public-boundary regressions for hostile scalar and container controls without changing ATA information, selection, or scoring arithmetic.

#### Fleiss kappa control trust boundary

- Hardened the public Fleiss/Conger kappa control boundary so explicit category counts and exact-mode selection are validated without executing caller-defined integer, index, or truthiness callbacks before ratings materialization or compiled-core discovery.
- Preserved genuine Python/NumPy scalar compatibility, capped explicit and inferred category counts at the Rust contract maximum of 10,000, and kept all agreement arithmetic Rust-owned.

#### Selection utility numeric trust boundary

- Hardened classical selection-utility and Taylor-Russell scalar controls so booleans, non-real objects, and non-finite values fail with package-owned validation before compiled Rust discovery.
- Prevented arbitrary caller-defined `__float__` callbacks from executing during public control marshalling while preserving genuine Python/NumPy real scalar compatibility and keeping all BCG, Naylor-Shine, and Taylor-Russell arithmetic Rust-owned.
- Normalized exact built-in integers outside the representable float range to the same package-owned validation error instead of leaking `OverflowError`.

#### Essay report title trust boundary

- Hardened score, validation-evidence, and facets-calibration essay HTML renderers so caller-supplied titles admit only exact built-in strings, rejecting caller-controlled `str` subclasses before overridden text callbacks such as `strip()` or HTML-escaping operations can execute.
- Added hostile-string-subclass regressions that prove all three public renderers reject before callback execution or artifact creation; scoring, calibration estimation, and psychometric arithmetic remain unchanged.

#### Factor-rotation semantic control trust boundary

- Reject caller-defined criterion/policy strings and boolean, integer, or real conversion protocols before factor-rotation Rust-core discovery across direct rotation, criterion-gradient, and empirical criterion-selection APIs.
- Preserve exact built-in and supported concrete NumPy scalar controls while keeping rotation objectives, gradients, multi-start optimization, convergence, bootstrap diagnostics, policy scoring, and criterion selection arithmetic Rust-owned.

#### Harden rubric text schema callback safety

- Harden rubric, item-blueprint, and shared scoring text/identifier schema admission so caller-defined `str` subclasses fail closed before any overridable text callback executes, while preserving normalization for exact built-in strings.
- Apply the same exact-built-in-string admission to item-bank evidence enums so lifecycle evidence cannot dispatch caller-defined equality or hash callbacks during enum lookup.

#### Restore semantic essay table row headers

- Mark the identity axis of governed essay facets-calibration and validation-evidence tables with explicit `<th scope="row">` semantics. Task, rater, respondent, category/iteration, and validation-metric identities now remain programmatically associated with their row while numerical scoring and calibration arithmetic remain unchanged.
- Preserve complete table and canonical-JSON evidence when standalone reports are printed or exported to PDF by removing screen-only scroll clipping and the JSON height cap in print media.

#### Harden generic diagnostics report title callback boundary

- Reject caller-defined `str` subclasses at the public generic diagnostics-report title boundary before truth-value or HTML-escaping callbacks can run, while preserving `None` and an empty exact built-in string as requests for the report-type default title.

#### Bound GPU smoke package provisioning

- Bound Vulkan package index and installation network/lock waits with explicit APT request, retry, lock, and whole-command deadlines so a hosted-runner mirror stall fails with actionable provisioning evidence instead of consuming the full GPU job timeout.
- Route the GPU smoke job through an isolated deb822 source list backed by the canonical Ubuntu archive and security endpoints, preventing the hosted runner's `mirror+file` registry from repeatedly selecting a black-holed Azure mirror for package payloads after metadata fallback.
- Preserve the existing llvmpipe Vulkan adapter proof and explicit CPU/GPU parity test; this changes CI provisioning reliability only, not production numerical behavior.

### Security

#### Seal ATA content-string callback admission

- Reject caller-defined string subclasses at Automated Test Assembly content-label and content-constraint-key validation boundaries before package-triggered text conversion callbacks or psychometric scoring can run, while preserving exact built-in and NumPy string scalar support.

#### Rubric generation text callback safety

- Reject caller-defined `str` subclasses at source-content, generation-contract JSON, candidate-parser JSON, static-fixture response, and live provider-output admission boundaries before caller-overridable text operations can execute.
- Preserve built-in string behavior, exact source whitespace and digests, redacted provider failures, deterministic generation provenance, and the existing Rust-owned psychometric/statistical computation boundary.

#### Compensatory 2PL control trust hardening

- Validate and normalize `q`, `estimate_corr`, `max_iter`, `tol`, `xi_points`, and `xi_seed` before response-array materialization or native-core discovery, rejecting caller-defined scalar subclasses and arbitrary conversion/truth-value providers without executing their callbacks.
- Preserve documented built-in and concrete NumPy scalar compatibility, Gauss-Hermite node choices, positive finite tolerance, iteration and QMC/MC point limits, and the full unsigned-64 integration-seed domain while passing only normalized built-in primitives to Rust.
- Keep compensatory 2PL likelihood, integration, ECM correlation estimation, convergence, and EAP arithmetic unchanged in the Rust core.

#### Testlet input trust hardening

- Validate testlet estimator semantic controls before materializing caller-owned response or testlet arrays, so invalid controls fail without executing array protocols or reaching native-core discovery.
- Reject complex and object/string-like response storage before real-valued narrowing, preventing imaginary response evidence from being discarded and preventing caller-controlled per-element numeric conversion during admission.
- Preserve the existing 0/1/NaN response contract, testlet identifiers, resource bounds, and Rust-owned marginal-ML EM, quadrature, convergence, and local-dependence arithmetic.

#### Parallel-analysis control trust hardening

- Validate `n_iterations`, `centile`, and `seed` before native-core discovery, accepting only exact built-in integers and supported concrete NumPy integer scalars while rejecting booleans, `np.bool_`, caller-defined subclasses, and conversion providers without executing their callbacks. Workspace and `u64` seed limits fail at the same pre-discovery boundary.
- Normalize nonnumeric `data` conversion failures to a package-owned `ValueError` before native-core discovery while preserving dimensionality and workspace validation for successfully converted arrays.
- Preserve the existing positive-iteration, centile `0..99`, Rust `u64` seed, and 128 MiB random-benchmark workspace limits without changing Rust-owned Horn/Glorfeld factor-retention arithmetic.

#### Harden validation-policy scalar trust boundaries

- Reject caller-defined string and numeric subclasses at `ValidationPolicy` construction before `strip`, numeric conversion, or comparison callbacks can execute.
- Normalize only exact built-in and package-trusted NumPy real scalar identities for scoring-policy thresholds while preserving the existing closed `0..1` domains and Rust-owned pass/fail arithmetic.
- Require an exact built-in integer for `min_subgroup_n` before range comparison and preserve the existing `rust_kwargs()` payload contract.

## [0.8.0] - 2026-08-17


### Fixed

- Public spatial/marginal MMLE now wraps a version-matched Rust `fit_marginal` keyword `TypeError` as the package-owned ABI `RuntimeError`, so a stale native signature cannot leak past `MARGINAL_CAPABILITY_VERSION = 1` or fall back to NumPy production arithmetic.
- Public multigroup M2 fails closed without the compiled Rust core and delegates target/null projected quadratic forms to native `projected_m2`.
- Public conditional-Rasch M2 fails closed without the compiled Rust core and delegates every result field to the native `m2_cmle_rasch_stat` entrypoint.
- Reject ambiguous LLM-judge JSON with duplicate keys or unexpected top-level fields; require the exact mode-specific schema including advisory `accepted`.
- Require compiled Rust ownership for public `s_x2` and `person_fit`, including prior-mean S-X² dispatch, with fail-closed errors when the core is missing.
- Validate parallel-analysis integer controls and bound random-eigenvalue workspace before Rust dispatch.
- Cap LLM-judge response JSON nesting at 32 levels before parse to prevent recursive-object resource exhaustion.
- Public fixed-form `assemble_test_form` delegates greedy maximum-information selection and content-feasibility look-ahead to the Rust core (`assemble_test_form_greedy`).
- Public fixed-anchor `link_fixed_item_parameters` delegates affine scale/shift estimation and parameter transformation to the Rust core.
- Public `observed_information` and `second_order_test` delegate Hessian assembly and eigenvalue diagnostics to the Rust core.
- Public CAT `item_information` and `select_cat_item` delegate Fisher information and maximum-information ranking to the Rust core.
- Bound top-1 CSR loser streams and enforce the shared ranking CSR byte ceiling with stable non-reflective iteration errors.
- Validate ATA content-constraint maps, exposure counts, seed, and exposure_max as admitted types before item-information evaluation, rejecting hostile conversion callbacks while preserving accepted string keys and exact integers.
- Cap LSR ranking CSR geometric growth under the live byte budget and stream validated item indices without list→uint64 temporaries beside the handoff arrays.
- Closed the Python-to-Rust equivalent-groups equating control boundary:
  method and explicit score-ceiling controls now reject arbitrary objects
  instead of invoking caller-defined ``__str__``/``__int__`` methods.
  Built-in Python integers and NumPy integer scalars remain supported. Added
  fail-closed regression tests and APA 7th doctoring for the validation
  contract.

### Added

#### Paired rating-range evidence

- Added a Rust-owned paired automated/reference rating diagnostic for observed category endpoints, distinct category use, span, empirical dispersion, relative ratios, endpoint gaps, narrower-support evidence, and a conservative central-tendency signal.
- Added a thin PyO3/Python product path that delegates numerical statistics to `mlsirm-core` and keeps descriptive range-use evidence separate from agreement, rater severity, and future generalized many-facet range-restriction parameters.
- Added fail-closed input/degenerate-reference behavior and APA 7 doctoring for automated essay-scoring validation.

#### Multilevel, multiple-membership, and longitudinal design contracts

- Added a provider-neutral `fast_mlsirm.multilevel` contract namespace for one-hot nesting, cross-classified designs, weighted multiple membership, multiple-membership multiple-classification, and repeated longitudinal occasions. Every contextual edge names an explicit `context_dimension_id` and `context_id`; schema 1.0 never infers or invents a random-effect family from a context label.
- Added independent weight normalization within every observation-by-context-dimension group, required coverage of every declared context dimension, dimension-scoped duplicate and context identities, exact per-dimension count/weight serialization, and assignment-revision fingerprints bound to the precise observation, dimension, context, and weight.
- Added deterministic SHA-256 identities, descriptive 128-bit public handles, child-artifact replay protection, bounded and callback-safe collection handling, strict respondent-level occasion ordering, and source-text-free serialization.
- Added separate random-intercept/slope and discrete occasion-step stationary AR(1) state specifications with independently controlled lagged-response dependence. Irregular millisecond offsets remain provenance only; continuous-time or interval-adjusted transitions require a later explicit Rust contract.
- Added realistic contract and adversarial tests, an MSA RFC, staged implementation plan, and APA 7 doctoring while reserving all likelihood, integration, optimization, uncertainty, multithreading, GPU work, and true-parameter recovery for future Rust cores.

#### Rust-owned sparse weighted contextual-effects predictor

- Added `mlsirm_core::multilevel::weighted_contextual_effect`: the contextual term `sum_h w_ph u_h` of the multilevel linear predictor (Browne, Goldstein, & Rasbash, 2001) over a sparse CSR-style cross-classified multiple-membership design. Ordinary nesting is the one-hot special case (`w_ph = 1` for exactly one edge per dimension), not a separate code path.
- Deterministic regardless of edge order within an observation, observation order, or worker count: each row is summed in ascending context-index order and rows are independent, backed by a bounded manual `std::thread::scope` worker pool (no new dependency).
- Added one-hot-nesting-parity, weighted-membership, cross-classified-dimension, permutation-invariance (edge order and row order), and worker-count-determinism Rust unit tests, plus fail-closed validation of malformed CSR offsets, out-of-range context indices, and non-finite/negative weights.
- Added a `_multilevel_core` PyO3 extension module (dual-`PyInit_*` pattern, matching bifactor/rotation/rating-range) exposing `weighted_contextual_effect` as a marshal-only numpy binding, plus `fast_mlsirm.multilevel.weighted_contextual_effect`, which marshals a validated `ContextMembershipDesign` and a per-context effect mapping into the Rust call and back.
- Reserves the Bayesian/MCMC estimation of the random effects `u_h` themselves, longitudinal state transitions, uncertainty, GPU batch path, and fairness/DIF work for the later staged PRs in issue #565.

#### Exact-value tooltips and print optimization for essay HTML reports

- Supplemental native `title` tooltips exposing unrounded exact float representations on formatted cells in essay score HTML reports.
- CSS `@media print` rules enforcing black-on-white text, hiding interactive skip links, and avoiding awkward page breaks for print and PDF exports.

#### Governed factor-retention evidence contract

- Added a provider-neutral `fast_mlsirm.factor_retention` contract that records already-computed candidate counts from supported retention methods, rejects duplicate method evidence, and reports `consensus`, `disagreement`, or `insufficient_evidence` without forcing a winner when methods disagree.
- Added deterministic conservative candidate ranges, a fixed transport ceiling, closed method identities, complete fail-closed tests, and scientific doctoring while keeping factor-retention and structural model-selection arithmetic Rust-owned and separate.

#### Relation-safe structural model comparison contract

- Added a typed structural measurement-model relation contract that keeps factor
  retention separate from structural model choice and classifies model pairs
  from explicit parameter-space, boundary, constraint, overlap, and formal
  distinguishability facts rather than model names.
- Restricted regular likelihood-ratio procedures to regular nesting, routed
  boundary/unidentified/nonlinear restrictions to conservative bootstrap LR,
  required formal Vuong distinguishability before non-nested selection, and
  returned explicit no-selection or unknown states instead of forcing a winner.
- Added fail-closed contradiction, exact-Boolean, boundary-precedence, and
  procedure-routing tests plus APA 7 doctoring; no comparison statistic or
  estimator is introduced by this contract slice.

#### Leakage-safe model-validation units

- Added a provider-neutral `fast_mlsirm.model_validation` contract that requires model-selection validation to declare a scientific generalization unit rather than silently splitting response cells.
- Added group-partition validation that rejects one declared person/system, query/testlet, rater/family, domain/language, cluster/context, or temporal group appearing across folds.
- Added temporal-forward validation that requires an explicit temporal-period unit and rejects any window whose latest training period overlaps or follows the earliest validation period, preventing look-ahead while keeping calendar interpretation caller-owned.
- Kept predictive scoring, bootstrap statistics, likelihoods, and other result-affecting psychometric arithmetic outside this Python validation/orchestration boundary and under Rust ownership.

#### Governed post-pilot item-bank lifecycle

- Add a factory-sealed, content-addressed post-pilot item-bank lifecycle that requires exact calibration, item-fit, DIF, information, approval, drift, suspension, and retirement evidence before an item can advance through `piloting`, `calibrated`, `approved`, `active`, `suspended`, reactivated, or terminal `retired` states.
- Preserve policy criticality independently of psychometric discrimination, require use-specific approval, link every successor to the exact previous record fingerprint, and retain only source-text-free evidence identities while leaving numerical calibration and item-bank arithmetic Rust-owned.
- Keep tenancy, authorization, identity mapping, persistence, encryption, retention, deletion, human governance, provider SDKs, new estimators, version bumps, and releases outside this reusable-core slice.

#### Governed item-bank lifecycle contracts

- Add immutable, content-addressed item-bank lifecycle and release contracts that bind generated items to exact rubric, blueprint, generation, audit, screening, pilot, calibration, approval, retirement, and linking evidence without adding hosted persistence or new numerical ownership.

#### Governed RAG scoring request

- Reference-free RAG scoring request adapter with privacy-preserving identity channels and fail-closed rejection of raw system configuration content.

#### Governed automated-scoring validation threshold policy

- Added `ValidationPolicy` and verdict `policy_id`/`policy_version` fields so
  automated-scoring acceptance gates identify their governing policy.
- Marshaled policy thresholds (`qwk_min`, `pearson_r_min`, `degradation_max`,
  overall/subgroup SMD, `min_subgroup_n`) into the Rust `validate_scoring`
  decision owner instead of hard-coding Williamson high-stakes cutoffs only in
  Python.

#### Governed RAG perturbation anchors

- Added source-free, content-addressed reference-free RAG perturbation anchors with finite preregistered construct/direction semantics for unsupported claims, contradictions, irrelevant context, required-evidence removal, citation swaps, semantic paraphrases, style-only rewrites, and unanswerable queries.
- Require canonical governed baseline and perturbed `ScoringRequest` values, reject unrelated or mixed-axis pairs, and bind each anchor to exact perturbation specification/run fingerprints while serializing only source-free identities.
- Distinguish literature-aligned constructs from package-owned model-design hypotheses. Every expected direction remains a validation hypothesis, not a claim that the cited papers established the exact perturbation, that an observed system actually changed, or that an evaluator is ground truth.

#### Essay facets synthetic recovery evidence

- Deterministic governed synthetic recovery coverage for scoring-facets MFRM/RSM
  recovery against true injected rater/item effects (issue #397).

#### Architecture baseline documentation

- Root `ARCHITECTURE.md` describing layered Rust-primary numeric core, Python
  orchestration, multilevel/multigroup population structures, modular MSA
  stance, security/compliance posture, and recovery-oriented test strategy with
  APA 7th citations.
- `docs/doctoring/architecture_baseline.md` binding the architecture claims to
  the psychometrics literature used by the product.

#### Enterprise criterion-level observation adapter

- Added `build_enterprise_issue_score_observation`, which compiles exact
  request-bound enterprise evidence into the existing shared criterion-level
  `ScoreObservation` contract without introducing a parallel observation schema.
- Added fail-closed request-provenance replay, evidence-subset validation,
  deterministic evidence ordering, managed issue/evidence fingerprints, and
  supporting, counter, and context evidence counts without retaining source text.
- Required supporting evidence for every non-abstained enterprise observation and
  explicit counterevidence representation whenever the issue declares
  counterevidence; insufficient evidence remains an abstention rather than a low
  score.
- Added deterministic provenance, order-invariance, abstention, terminal-state,
  adversarial metadata, evidence-binding, shared-contract delegation, and
  statement/branch coverage tests for the next issue #404 vertical slice.

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
  Provider-owned records are reconstructed as fresh canonical instances,
  manually supplied offsets share the enterprise source-character bound, and
  deterministic candidate producers stop at the configured limit plus one rather
  than exhausting unexpectedly prolific iterators.
- Added immutable content-addressed records that compile exact occurrences into
  the existing directly stated `EvidenceSpanRecord` boundary without adding a
  parallel scoring, observation, result, or engine schema.
- Added deterministic, protocol, privacy, security, exact-decimal, offset,
  ordering-invariance, metamorphic sentiment-independence, and fail-closed tests,
  plus APA 7th standards traceability and conservative interpretation limits.

#### Governed enterprise issue calibration bundle

- Added `build_enterprise_issue_facets_calibration_bundle()` as a bounded,
  fail-closed assembler from exact enterprise issue scoring execution tuples into
  the existing shared `ScoringFacetsCalibrationBundle` contract.
- Reused the complete issue-owned provenance replay for every execution and
  delegated criterion separation, task-revision and rater identity, category
  support, record budgets, and connectedness to the existing shared bundle
  builder without adding a competing enterprise schema.
- Added deterministic order-invariance, exact tuple-shape, delegation,
  resource-bound, public-export, and connected-design tests while preserving
  Rust-only psychometric arithmetic and conservative validity, fairness, and
  causal limits for issue #404.

#### Enterprise issue evidence contracts

- Added immutable content-addressed enterprise source, evidence-span, atomic-issue,
  counterevidence, stakeholder-perspective, and candidate-intervention contracts.
- Added deterministic compilation into the shared scoring `EvidenceReference`
  boundary while preserving facts, inferences, counterevidence, ambiguities, and
  stakeholder value judgments as distinct epistemic roles.
- Added fail-closed provenance, source-revision, sensitive-metadata, and ordering
  tests for the first issue #404 domain-adapter slice.

#### Governed enterprise issue facets calibration reports

- Added `fit_enterprise_issue_facets_calibration_reports()` as a bounded governed
  workflow from exact enterprise issue scoring executions to one existing shared
  `ScoringFacetsCalibrationReport` per criterion.
- Reused the enterprise provenance replay and shared calibration bundle assembler,
  then delegated every criterion design to the existing Rust-backed shared report
  helper without adding an enterprise-specific fit, report schema, or statistical
  arithmetic.
- Added package-managed bundle, design, and criterion report provenance,
  deterministic execution-order invariance, one-time review-trigger
  normalization, source-free caller metadata, and reserved-key rejection.
- Added fail-closed batch validation of every derived report identifier before any
  Rust estimator delegation, preventing an overlong prefix-and-criterion
  combination from producing a partially fitted report tuple.
- Added a realistic connected two-issue, two-task-revision, two-rater-family,
  two-criterion Rust fit, complete orchestration and privacy tests, public
  documentation, and APA 7th scientific and governance traceability.
- Aligned shared report and HTML replay validation with the Rust estimator's
  nonconverged trace contract: `n_iter` optimization iterations may be followed
  by one retained terminal post-update likelihood evaluation.

#### Governed enterprise issue many-facet handoff

- Added `build_enterprise_issue_facets_rating_records()` as a fail-closed replay
  boundary from exact enterprise issue scoring executions into the existing
  shared `ScoringFacetsRatingRecord` contract.
- Replayed atomic issue, issue-content, respondent, response-revision,
  request-bound evidence, counterevidence, and package-managed observation
  provenance before delegating record projection to the shared calibration
  builder.
- Preserved abstention as a terminal missing rating, separate analytic criteria,
  Rust-only psychometric arithmetic, deterministic order invariance, complete
  statement and branch tests, and conservative validity and causal limits for
  issue #404.

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

#### Enterprise semantic issue provider boundary

- Added runtime-checkable `EnterpriseAtomicIssueExtractor` and
  `extract_enterprise_atomic_issues` as a provider-neutral, provider-SDK-free trust
  boundary that returns the existing canonical `AtomicIssueRecord` contract.
- Added bounded exact source-packet replay, UTF-8 and Python code-point span
  verification, fresh nested issue/evidence/counterevidence reconstruction,
  deterministic ordering, duplicate and overlap rejection, and redacted provider
  failures without retaining raw enterprise text.
- Added `StaticEnterpriseIssueExtractor` as an offline fixture and integration
  adapter that performs no NLP, sentiment analysis, issue discovery, scoring,
  ranking, utility, or causal arithmetic.
- Added deterministic order-invariance, all-assertion-kind preservation,
  malicious provider, source mutation, span replay, subclass, privacy, prolific
  collection, duplicate identity, overlap, and complete statement/branch coverage
  tests for the next issue #404 workflow slice.

#### Accessible standalone essay facets-calibration artifacts

- Added `render_essay_facets_calibration_report_html`, which replay-verifies one governed `EssayFacetsCalibrationReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, design, assessment, rubric, construct, occasion, criterion, respondent, task-revision, rater-engine, category, estimate, convergence, connectedness, iteration, and review-trigger evidence through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- A restrictive meta-delivered Content Security Policy and output encoding reduce injection impact; convergence and connectedness remain integrity prerequisites and do not establish model fit, reliability, fairness, scorer interchangeability, construct validity, global optimality, or deployment authorization.
- Confined report publication to a canonical caller-approved output directory, with current-working-directory defaults, traversal and absolute-escape rejection, existing symlink-parent resolution, post-creation parent revalidation, and fail-closed non-directory roots.

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

#### Governance index

- `docs/GOVERNANCE_INDEX.md` consolidating ADR index, threat-model summary,
  test strategy, operability, traceability matrix, and component UML with
  APA 7th doctoring links.

#### Hourly bounded review-repair caller

- Added a schedule-only fast-mlsirm caller that runs at minute 37 every hour and
  delegates to one immutable organization-owned review-repair workflow.
- Bounded each run to one new repair dispatch, one-hour same-head retries,
  protected `main`, non-cancelling product-level single-flight concurrency, and
  explicit scheduler credentials without direct model secrets or inherited
  secrets. A delayed next heartbeat does not discard an in-flight bounded scan;
  exact-head retry and single-writer controls remain owned by the central worker.
- Kept the workflow-generated `GITHUB_TOKEN` read-only at both workflow and call
  job scope; cross-repository mutation requires an explicitly forwarded
  established scheduler credential and fails closed when none is available.
- Added permanent caller-contract tests and APA 7th doctoring for default-branch
  activation, immutable reusable-workflow source, failure behavior, rollback,
  and the NVIDIA NIM control-plane boundary.

#### Governed RAG facets calibration

- Added the governed RAG facets calibration adapter.
- The adapter reuses the existing MFRM rater-severity/threshold calibration
  machinery (Linacre, 1989; Eckes, 2015; Bock & Aitkin, 1981; Andrich, 1978)
  for RAG evaluation executions. It does not introduce a new psychometric
  estimator or rely on legacy evaluation package implementations (e.g.
  RAGAS-style tooling) as the source of psychometric validity; all
  likelihood/threshold arithmetic is delegated to the existing Rust-backed MFRM
  fit grounded in the primary many-facet Rasch measurement literature. Full
  citations are in `docs/scoring_facets_calibration_handoff.md`.

#### Supplemental exact-value report tooltips

- Added native `title` tooltips to finite floating-point metric cards, decorative
  bar labels, and diagnostic table cells so pointer users can inspect the
  unrounded Python float representation when the visible report uses compact
  significant-digit formatting.
- Preserved the existing accessible exact-value disclosure and JSON/CSV exports as
  the authoritative keyboard, touch, and assistive-technology paths; native title
  tooltips are supplemental and are not treated as an accessibility substitute.
- Added deterministic metric, chart, table, finite-value, non-finite-value, and
  non-float tests for the report tooltip contract.

#### Shared scoring-facets calibration report names

- Added `fast_mlsirm.scoring.calibration_reporting` as the domain-neutral import
  path for the canonical scoring-facets calibration report, preserving exact
  object identity and existing essay ABI and wire contracts.

### Changed

#### Explicit criterion row headers in governed HTML reports

- Replaced positional first-column row-header inference with an explicit
  zero-based `row_header_column` contract in the governed essay score renderer.
- Emit `<th scope="row">` only for criterion identifiers; evidence-reference
  values remain data cells, and malformed header/row widths fail closed.
- Added complete-artifact parsing tests that verify table semantics and exact
  canonical JSON reconstruction, plus APA 7th WCAG 2.2 doctoring.

#### Bounded fit-statistics fallback buffers

- Reused one owned NumPy squared-residual buffer for fallback infit/outfit calculations, retained the infit numerator before in-place division, and used a Boolean `where=` reduction for the variance denominator without a full numeric mask copy.
- Added deterministic sparse-missingness and clipped-probability parity contracts plus an environment-specific benchmark while retaining Rust as the production fit-statistics backend.

#### Essay-validation empty-state status semantics

- Marked empty identifier evidence in standalone essay-validation HTML reports as a WAI-ARIA `status` region with explicit `aria-atomic="true"`, while preserving visible text and avoiding focus movement.
- Added a deterministic regression for the exact status markup and documented the interoperability boundary: live-region semantics improve assistive-technology exposure for status updates, but a pre-populated static report is not claimed to trigger an initial announcement.

#### Diagnostics-report numeric alignment and motion cleanup

- Applied tabular numeral styling to standalone diagnostics-report body text so numeric values can align more consistently when the selected font supports equal-width figures.
- Removed obsolete opacity transitions from bar rows and table rows while preserving the active table-row background hover cue and the existing reduced-motion override.
- Added a rendered-report regression that pins the numeric-style declaration, transition cleanup, hover cue, and reduced-motion contract without changing report data, score semantics, or exported exact values.

#### Python 3.14 CI compatibility

- Expanded the full Python CI job from a single CPython 3.12 runtime to a fail-slow CPython 3.12 and 3.14 matrix while preserving the existing Rust/PyO3 build, Rust-primary backend verification, package, GPU, fuzz, and security gates.
- Added a deterministic CI contract that requires Python 3.14 to execute the same complete pytest suite rather than a reduced compatibility smoke path.
- Added a non-matrix `python` aggregate job so branch-protection's required check context named `python` still receives a single SUCCESS/FAILURE after every matrix leg finishes.

#### Rust-owned CAT ability estimation

- Moved public CAT MLE, EAP, and ability-standard-error arithmetic from the
  Python wrapper into the compiled Rust scoring core, retaining Python only for
  validation, marshalling, and adaptive-test policy.
- Added bounded Newton MLE, prior-centred grid EAP, scoped per-dimension Rust
  workers, and device-aware information reduction with explicit handling for
  all-identical response patterns and unadministered dimensions.
- Added sentinel delegation tests, Rust edge-case tests, seeded true-trait
  recovery tests, score-equation checks, and APA 7th CAT doctoring.

#### Acquisition readiness gate

- Sales readiness no longer fabricates a KRW 2B contract value by default; deal scenarios must be supplied explicitly, and a generic `--require-acquisition-readiness` profile now activates buyer-packet, benchmark, release-evidence, procurement, PR-queue, and Figma validators without the legacy 20B file/token bundle.
- The legacy `--require-20b-product` profile remains a deprecated compatibility mode, while the manifest records generic readiness, compatibility mode, and transaction-scenario identity separately.

#### Automatic backend preserves Rust numerical ownership

- Changed `backend="auto"` so a missing compiled Rust core fails closed instead of silently selecting the independent NumPy reference implementation.
- Kept explicit `backend="numpy"` as an explicit reference/parity choice while preserving automatic Rust resolution and Rust CPU/GPU device fallback semantics.

#### Fit-statistics tail ownership

- Public `chi2_sf` and `benjamini_hochberg` now prefer the Rust core for ranking and tail arithmetic, with a pure-Python fallback only when the compiled core methods are unavailable.

#### Inference covariance ownership

- `vcov_from_hessian` and `standard_errors_from_vcov` now own inversion, pseudoinverse, and SE extraction in Rust.

#### Current support-policy version line

- Aligned the public security and support policies with the released `0.7.x` pre-1.0 package line instead of the obsolete `0.1.x` policy.
- Reframed support around released, documented public API and packaging behavior, preserved conservative high-stakes/certification/SLA boundaries, and clarified Rust-first production numerical ownership versus explicit reference/parity paths.
- Added a repository contract that derives the supported minor line from `pyproject.toml` so future package-version changes cannot silently leave `SECURITY.md` or `SUPPORT.md` stale.

#### Release serializes derived changelog aggregation

- Move fragment→`CHANGELOG.md` aggregate parity enforcement from ordinary feature CI into the immutable release-tag workflow so concurrent PRs no longer thrash a shared derived file while release publication still fail-closes on drift.

#### Direct Rust response-time iteration ceiling

- Enforce the package-wide `max_iter` ceiling (`1..=100_000`) inside the Rust
  lognormal response-time EM (`fit_rt_lognormal`) so direct PyO3 callers cannot
  bypass the Python `MAX_MAX_ITER` bound.
- Added a fail-closed ownership contract that rejects `MAX_MAX_ITER + 1` at the
  Rust boundary and documented the van der Linden (2007) speed model reference.

#### Governed automated-scoring validation threshold policy

- Default policy remains `williamson_high_stakes` v1.0 with the published
  high-stakes thresholds; invalid threshold ranges fail closed before Rust work.

#### ATA target-gain Rust ownership

- Moved result-affecting capped-shortfall target-information gains for automated
  test assembly from Python/NumPy into a bounded Rust PyO3 kernel.
- Kept Python responsible for validated candidate/content/exposure orchestration
  and deterministic tie breaking while the compiled path owns candidate gain
  arithmetic without candidate-by-point broadcast temporaries.
- Made the public PyO3 boundary reject wrong-dtype, non-array, non-contiguous,
  empty-matrix and overlong candidate-set inputs with stable package-owned
  `ValueError` messages before candidate/output allocation.
- Bounded candidate inputs to the item count represented by the information
  matrix and converted both candidate and result vectors with fallible reserve.
- Added direct Rust and installed-extension parity/ownership regression evidence.

#### Conditional Rasch M2 Rust ownership

- Public `m2_cmle_rasch()` and `m2(..., estimator="cmle")` fail closed without the compiled Rust core and delegate every result field to `m2_cmle_rasch_stat`.

#### Fail-closed IRT linking control values

- Validate `irt_link(method=...)` as an exact built-in string against the existing Rust `LinkMethod` vocabulary before loading or calling the native core.
- Validate `q_theta` as either an exact built-in Python integer or a genuine NumPy integer scalar before native-loader access; integer subclasses are rejected before caller-controlled `__int__`/representation callbacks can run.
- Reject hostile method objects, string subclasses, unsupported method identities, and hostile quadrature subclasses with package-owned `ValueError` evidence, while preserving trusted Rust-supported aliases and genuine NumPy integer quadrature scalars.
- Keep all IRT scale-linking coefficients, characteristic-curve criteria, optimization, convergence arithmetic, and quadrature generation behavior in their existing numerical owners; this change is limited to Python validation and marshalling.

#### Default LLM judge orchestration to adaptive auto mode

- `ContextualOrchestratorJudge` now defaults ordinary calls to contextual-orchestrator `auto` mode while preserving explicit `route` and `conduct` overrides and the fail-closed `contextual-orchestrator-contract-v1` adapter boundary.

#### MMLE theta calculation memory optimization

- Replaced the NumPy reference/fallback EAP expression `(posterior * nodes[None, :]).sum(axis=1)` with the algebraically equivalent matrix-vector product `posterior @ nodes`. This avoids constructing the explicit posterior-shaped broadcast product; NumPy may use optimized BLAS for matrix multiplication when available, while realized runtime remains dependent on array shape, layout, hardware, and the linked numerical library.

#### Canonical product and architecture documentation baseline

- Replaced the stale MVP-only PRD/TRD authority with canonical `docs/PRD.md` and `docs/TRD.md` requirements covering the current measurement, scoring, rubric/item-generation, model-selection, scientific-evidence, interoperability, security, lifecycle, and release boundaries.
- Added root `ARCHITECTURE.md`, a status-bearing ADR corpus, reviewable PlantUML component/sequence/state/deployment views, a logical reusable-domain ERD, and requirements/research traceability matrices.
- Added a canonical documentation authority index, explicit implementation-maturity/completeness matrix, and machine-checkable documentation contract so missing or stale PRD/TRD/ADR/UML/ERD/traceability/security artifacts remain visible release-maintenance debt rather than silently drifting.
- Added a reusable-core threat model covering provider/JSON replay, native/PyO3 input boundaries, resource and non-finite numerical failures, GPU evidence spoofing, supply-chain/self-modifying CI, credential separation, benchmark contamination, privacy/purpose limitation, and scientific-interpretation abuse while leaving hosted HTTP/session/tenant/database threats downstream.
- Added durable ADRs for converging future Rust-backed features on one canonical PyO3/public-export registry and for preserving legitimate sensitive-data linkage through purpose limitation and minimization rather than blanket masking that changes the measurement design.
- Extended requirements traceability with the conversation-wide invariants that human/LLM judges are fallible raters, correlation is not parameter recovery/absolute agreement, latent-space interaction follows substantive dimension/testlet/facet diagnosis, reference-free is not truth-free, and psychometric discrimination is not business or safety criticality.
- Explicitly deprecated the original narrow `docs/prd_trd_summary.md` as an authoritative requirements source while retaining its historical MLS2PLM MVP context.
- Defined the `fast-mlsirm-cjson-v1` fingerprint preimage, SHA-256 binding, null/ordering/Unicode/number rules, and cross-language normative vector instead of leaving canonical serialization as an interoperability assumption.
- Added the persistence-neutral `docs/uml/domain-public-contract.puml` view, indexed every UML source including the compatibility alias, modeled versioned calibration-design inputs as a many-to-many association, and made corrected quarantined items new immutable revisions.
- Added complete APA 7 research records and scope summaries for LLM-RUBRIC, AutoNuggetizer/TREC RAG, EvalGen, the 2025 AutoNuggetizer follow-up and 2026 reflective rubric research, plus NIST AI RMF governance inputs with explicit non-certification language.

#### Exact task-revision identity for scoring calibration

- Scoring-request wire schema `1.1` now requires an exact provider-neutral `task_revision_fingerprint` in addition to the logical task identifier. The fingerprint participates in request identity, is propagated by the essay adapter from the complete prompt fingerprint, and prevents changed task content from being silently pooled under one request or calibration item.
- Criterion-level many-facet handoffs now use exact task revisions as the Rust estimator item axis while retaining aligned logical task and task-family labels for audit. Duplicate cells, support, resource bounds, respondent–item connectedness, item–rater connectedness, and response provenance are all revision-indexed; one revision cannot be rebound to a different logical task or family.
- Added an explicit, fail-closed schema-`1.0` request migration that verifies canonical content, fingerprint, public handle, and the authoritative engine-policy projection; requires a caller-supplied task revision; preserves normalized caller metadata; and intentionally does not migrate legacy observations or results. Content identity prevents accidental pooling but does not establish cross-revision comparability, which still requires anchors, invariance/DIF, drift, and recovery evidence.

### Fixed

#### Operation-specific ignored Rust subprocess deadlines

- Added bounded operation-specific deadlines for Cargo metadata, ignored-test inventory, and long-running statistical-study execution in the ignored Rust shard runner while retaining the independent GitHub Actions job ceiling.
- Timed-out POSIX child groups now receive bounded `SIGTERM`-to-`SIGKILL` cleanup and machine-readable timeout evidence that omits command and captured child-output text; operator overrides remain constrained by per-operation minimum and maximum ranges.
- Added deterministic configuration, cleanup, redaction, command-routing, and cross-platform fallback contracts. Issue #555 remains open for the remaining repository subprocess operation classes.

#### Bounded marginal latent-distance workspaces

- Replaced the NumPy fallback's item-by-node-by-dimension covariate distance
  broadcast and cancellation-prone squared-norm identity with a
  coordinate-subtraction-first kernel that reuses one bounded two-dimensional
  float64 scratch buffer.
- Added a private 128 MiB distance-workspace ceiling, checked byte products,
  and pre-allocation gates for the pairwise output-plus-scratch peak, finite
  masks, and the intentional item-gradient workspace before latent nodes are
  built.
- Reused the governed helper in table construction, candidate predictors, the
  tau update, and the covariate update while preserving the Rust production
  backend and public model contracts.

#### Second-order diagnostics keep positive-definiteness semantics strict

- Rust-owned observed-information diagnostics now reject negative positive-definiteness tolerances instead of allowing callers to redefine a matrix with small negative eigenvalues as positive definite.
- Zero tolerance remains supported and preserves the strict requirement that every information eigenvalue be positive.
- Oversized second-order matrix dimensions whose square cannot be represented by `usize` now fail closed with a stable package error instead of overflowing dimension arithmetic.

#### S-X2 and person-fit Rust ownership fail-closed

- Public `s_x2()` and `person_fit()` require the compiled Rust core entrypoints and no longer fall back to Python/NumPy numerical implementations when the core or symbols are missing.
- `s_x2()` always dispatches trait `prior_mean` through the native S-X² entrypoint instead of selecting the Python reference path whenever a prior is supplied.

#### Parallel-analysis input and workspace bounds

- `parallel_analysis()` now rejects booleans, floats, strings, and caller-defined integer-conversion hooks for integer controls instead of silently coercing them before Rust dispatch.
- The public wrapper validates the Rust `u64` seed range and rejects oversized random-eigenvalue benchmark workspaces before PyO3 dispatch.
- `mlsirm-core` independently caps the random-eigenvalue simulation workspace at 128 MiB before allocation while preserving the existing Horn/paran numerical algorithm and deterministic RNG contract.

#### Subgroup validation evidence fails closed

- Automated-scoring subgroup SMD gates reject requested subgroups with fewer
  than two paired cases or zero human variance instead of silently skipping
  them and reporting a vacuous pass.

#### Serving bundle export requires Rust core

- `export_serving_bundle` fails closed when the compiled Rust core is unavailable
  instead of shipping incomplete bundles with null `eapsum_tables`.

#### Fail early for unimplemented estimator identities

- Restricted the public `FitConfig.estimator` vocabulary to the implemented `jmle` and `mmle` fitting paths, so unsupported `em` and `bayes` requests fail during configuration validation instead of entering a fitting path that later raises `NotImplementedError`.

#### Strict JSON artifact interoperability

- Governed JSON artifact writers now reject `NaN`, positive infinity, and negative infinity instead of emitting Python's non-standard JSON numeric extensions, preserving RFC 8259 interoperability and atomic publication failure.
- Non-finite serialization errors use a bounded package-owned message without reflecting the rejected artifact payload.

#### Node-rule fail-closed validation

- Public polytomous and 2PL fitters reject non-string integration-rule controls
  before importing the Rust core, without invoking caller ``__str__``/``__repr__``.

#### Bounded bifactor shape metadata inspection

- Replaced eager `tuple(shape)` materialization at the public bifactor scoreability boundary with package-owned bounded look-ahead.
- Matrix and uniqueness advertised shapes now stop after the minimum dimensionality proof, normalize ordinary caller iterator failures to stable non-reflective package errors, and preserve process-control signals.
- Kept accepted array marshalling, bifactor work ceilings, and all Rust-owned scoreability arithmetic unchanged; added fail-first resource/security regressions and primary Python 3.14 doctoring.

#### ATA content-label validation trust boundary

- Validate automated-test-assembly content-label shape and string element types before item-information evaluation, rejecting arbitrary object labels without invoking caller-controlled `__str__`/`__repr__` callbacks while preserving accepted Python/NumPy string labels and existing assembly numerics.

#### ATA semantic-range and exclusion preflight

- Reject negative or contradictory ATA content constraints, invalid exposure-map ranges, negative seeds, and non-integral or out-of-bank exclusions before item-information evaluation while preserving accepted Python/NumPy integer controls and assembly semantics.

#### Bound PR queue Git metadata lookup

- PR queue governance now bounds the `git rev-parse HEAD` subprocess and fails closed with a stable timeout error instead of allowing a hung local Git child to stall the evidence pipeline.

#### Release evidence Git metadata timeout

- Bound `git rev-parse` in the release evidence index builder with a fail-closed timeout.

#### ATA constraint-map validation trust boundary

- Validate ATA content-constraint keys/counts, exposure maps, seed, and exposure_max as admitted types before item-information evaluation, rejecting hostile string/integer conversion callbacks while preserving accepted Python/NumPy string keys and exact integers.

#### LSR ranking CSR live allocation budget

- Cap geometric growth of LSR/I-LSR ranking CSR `uint64` buffers so intermediate capacities never exceed the declared live CSR byte budget, and stream validated item indices without a list→`uint64` temporary beside the live payload.

#### Fit-statistics require compiled Rust core

- Public `chi2_sf` and `benjamini_hochberg` fail closed with a stable RuntimeError when the compiled Rust core is unavailable, preventing silent pure-Python numerical ownership.

#### CAT item information and selection Rust ownership

- Public `item_information` and `select_cat_item` now delegate Fisher information and maximum-information ranking to the compiled Rust core (`cat_item_information` / `cat_select_item`), reusing the frozen-bank information kernel already used by ability SE.

#### Non-finite inference uncertainty preserves scientific meaning

- Standard errors from covariance diagonals preserve `NaN` and infinite values instead of converting them into false zero uncertainty.
- `vcov_from_hessian` rejects non-finite observed-information entries with a stable finite-entry contract.

#### Top-1 CSR input bounds

- Bound top-1 loser streams to at most `n - 1` items, enforce the shared `MAX_RANKING_CSR_BYTES` ceiling on winner/loser/start `uint64` payloads, and normalize ordinary outer/inner iteration failures to stable non-reflective package errors while propagating process-control signals.

#### Observed-information Rust ownership

- Public `observed_information` assembles finite-difference Hessians in the Rust
  core from evaluated objective samples, and `second_order_test` eigenvalue
  diagnostics are Rust-owned.

#### Fixed-anchor linking Rust ownership

- Public `link_fixed_item_parameters` delegates affine scale/shift estimation and
  parameter transformation to the compiled Rust core, retaining Python for
  validation and evidence packaging only.

#### Fixed-form greedy assembly Rust ownership

- Public `assemble_test_form` delegates ordering, exclusion, and content-feasibility
  decisions to the compiled Rust core (`assemble_test_form_greedy`), keeping Python
  for validation and marshalling only.

#### JMLE Adam/L-BFGS Rust ownership

- Public JMLE `backend="rust"` routes Adam, L-BFGS, and `adam_lbfgs` sequencing
  through compiled `jmle_optimize` entrypoints so optimizer state updates no longer
  re-implement production arithmetic in Python loops.

#### LLM judge JSON nesting depth bound

- Cap raw LLM-judge response JSON nesting at 32 levels before `json.loads`, failing closed with `JudgeFormatError` so hostile recursive objects cannot expand into parser resource exhaustion.
- Keep valid shallow judge payloads accepted with the existing criterion/score contracts.

#### Fit-statistics infit/outfit and M2 fail closed

- Public `infit_outfit()` and ordinary `m2()` fail closed when the compiled Rust
  core or required entrypoints are missing, completing the residual ownership
  gaps from issue #627 after S-X² and person-fit hardening.

#### Documentation coverage vocabulary and shipped-capability matrix

- Align architecture documentation contracts with the protected-main maturity
  vocabulary and mark parallel-analysis control bounds and essay-report native
  dark-mode accents as ancestral after their integration.

#### Retire competing hourly review-repair caller

- Remove the repository-local hourly review-repair GitHub Actions caller so only
  the organization single-writer control plane schedules mutation loops, matching
  ADR-0013 continuous-execution governance after failed startup evidence for the
  local caller.

#### Observed-information work budget preflight

- Dense finite-difference `observed_information` preflights package-owned objective-call and fixed-width workspace budgets before the first objective evaluation and replaces the dense identity workspace with a reusable trial vector.

#### Rubric hostile iterable error redaction

- Rubric collection materialization fails closed on hostile iterable setup and iteration exceptions with package-owned messages, while preserving `MemoryError` resource signals.

#### Model-comparison hostile input redaction

- Model-comparison parameter counts and casewise iterables redact hostile conversion and iteration callback failures into stable package-owned `ValueError` messages while preserving `MemoryError`.

#### Multilevel hostile numeric callback rejection

- Multilevel membership weights and AR(1) coefficients now admit only exact
  built-in `int`/`float` scalars, rejecting booleans and caller-defined
  conversion hooks before contract arithmetic.

#### Scoring schema-version callback redaction

- Assessment schema-version validation now requires an exact built-in `str` matching the wire version, rejecting hostile string subclasses before equality work so callback messages cannot leak into contract errors.

#### Item-bank DIF applicability evidence

- Calibration transitions accept either DIF evidence or explicit
  `dif_not_applicable` evidence, forbid both at once, and keep other lifecycle
  gates unchanged.

#### Serving redundant parameter integrity

- Serving-bundle validation fails closed when exported redundant slope/distance-weight
  fields contradict canonical log-scale parameters, and admits only exact built-in
  numeric scalars so hostile conversion hooks cannot execute during load/score.

#### Factor-retention iterable error redaction

- Governed factor-retention evidence now converts hostile iterator-construction
  and iteration callback failures into stable package-owned validation errors
  without exposing caller-controlled exception text or chained causes.
- Explicit `MemoryError`, duplicate-method precedence, deterministic ordering,
  decision semantics, and the bounded closed-method evidence contract remain
  unchanged.

#### Multigroup M2 Rust projection ownership

- Public `m2_multigroup` fails closed without the compiled Rust core and delegates target/null projected M2 quadratic forms to `projected_m2`.

#### Report pointer focus correction

- Suppress the browser's default outline for pointer-focused report content while preserving the explicit keyboard-visible focus treatment.

#### Exploratory model factor-count callback safety

- Accept only exact built-in Python integers and genuine NumPy integer scalar types for exploratory factor counts, rejecting caller-defined integer subclasses before conversion callbacks can execute while preserving the existing positive-factor and multidimensional-support contracts.

#### G-theory NumPy scalar trust hardening

- Require exact package-supported NumPy integer and floating scalar classes for G-theory public numeric controls, rejecting caller-defined subclasses even when they spoof NumPy module metadata before any conversion callback can execute.

#### Mixture IRT control callback safety

- Validate mixture-model controls before native-core discovery, accept only exact built-in or supported genuine NumPy scalar identities, preserve the Rust binding's existing model aliases and tolerance semantics, and reject hostile scalar subclasses before conversion or representation callbacks can execute.

#### Continuous-response-model control callback safety

- Validate CRM quadrature, iteration, and tolerance controls before native-core discovery; accept only exact built-in or supported genuine NumPy scalar identities; preserve the Rust quadrature domain and convergence tolerance contract; and reject hostile scalar subclasses before caller-controlled conversion, comparison, ufunc, or representation callbacks can execute.

#### Paired rating-range category control hardening

- Require exact built-in or package-supported genuine NumPy integer scalar identities for `paired_rating_range_evidence(..., category_count=...)`, rejecting caller-defined subclasses before conversion, type-hash/equality, representation, or Rust-dispatch callbacks can execute.

#### G-theory pilot control callback boundary

- Hardened the generated-item G-theory pilot handoff so D-study sizes and mastery-cut controls accept only exact built-in or genuine supported NumPy scalar identities.
- Rejected caller-defined numeric and protocol subclasses before conversion, representation, hashing, or equality callbacks while preserving existing bounds and Rust-owned G-theory arithmetic.

#### Testlet pilot control callback boundary

- Hardened the generated-item testlet pilot handoff so model and execution controls establish exact trusted built-in or supported NumPy scalar identities before normalization or conversion.
- Rejected caller-defined protocol, numeric, and string subclasses without callback dispatch while preserving existing limits and Rust-owned testlet arithmetic.

#### Harden governed scoring execution integer boundaries

- Reject caller-defined integer coercion at governed scoring request, observation, and result controls before any `__index__` callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError` semantics.

#### Bounded hourly PR queue capture

- Split hourly open-PR identity enumeration from per-PR nested evidence capture so large queues no longer exceed GitHub GraphQL resource limits or publish a false zero-PR snapshot.
- Preserve fail-closed review, merge-state, label, changed-file, body, history, and exact default-branch evidence while excluding pull requests that close during capture.
- Fail closed when an open-PR detail payload omits required classification fields instead of promoting partial queue evidence.

#### Rubric integer control callback boundary

- Hardened rubric and blueprint integer normalization so exact built-in integers and genuine supported NumPy integer scalars remain compatible while caller-defined integer/protocol objects are rejected before executable conversion callbacks.
- Preserved the existing score, item-count, replicate-index, seed, and unsigned-64 bounds without changing psychometric arithmetic or Rust numerical ownership.

#### Harden essay adapter integer boundaries

- Reject caller-defined integer coercion across essay prompt limits, submission counts, and evidence offsets before any conversion callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError` semantics.

#### Harden enterprise evidence integer boundaries

- Reject caller-defined integer coercion for enterprise source character counts and evidence offsets before any conversion callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility, nonempty-span semantics, and existing bounded `AssessmentSpecError` behavior.

#### Harden observed-score equating control boundaries

- Validate NEAT, log-linear, kernel, and standard-error semantic controls before Rust discovery, rejecting executable coercion and comparison providers while preserving exact built-in primitives, genuine NumPy numeric scalars, documented Rust aliases, and Rust ownership of all equating arithmetic.

#### Reject executable ICC semantic controls

- Validate ICC model/type/unit choices and r0/confidence controls before native-core discovery or ratings materialization, reject caller-defined conversion/comparison protocol providers and scalar subclasses without executing their callbacks, preserve the Rust parameter ranges, and normalize trusted NumPy real scalars to exact Python floats before Rust dispatch.

#### Validate DETECT inputs before native discovery

- Validate and marshal DETECT and DIMTEST public response/partition inputs before compiled-core discovery, so rejected requests remain package-owned validation failures without crossing the native-loader boundary while all result-affecting dimensionality arithmetic remains Rust-owned.

#### Harden Rasch CML public controls

- Validate Rasch CML and Andersen LR response/group inputs plus trusted iteration/tolerance controls before compiled-core discovery, rejecting caller-defined scalar coercion while preserving genuine NumPy scalar compatibility and Rust-owned conditional-likelihood arithmetic.

#### Validate subscore inputs before native discovery

- Validate Haberman subscore response and partition inputs before compiled-core discovery, keeping rejected requests inside package-owned validation while preserving all PRMSE, reliability, covariance, disattenuation and added-value arithmetic in Rust.

#### Plausible-value serving control safety

- Validate and normalize public plausible-value `n_draws`, `seed`, and `device` controls before compiled-core discovery.
- Bound `seed` to the Rust/PyO3 `u64` contract, keep `n_draws` within the existing serving limit, and constrain device selection to `cpu`, `gpu`, or `auto`.
- Reject booleans, caller-defined integer/string subclasses, arbitrary coercion providers, and hostile scalar metaclasses without executing their conversion, hashing, or equality callbacks.
- Preserve exact supported NumPy integer scalar compatibility by admitting trusted scalar types through identity-only comparisons and marshalling them once to built-in integers.
Closes #914.

#### Secondary extension loader concurrency

- Serialize cache inspection and native initialization for the ATA, bifactor, multilevel, paired rating-range, and rotation secondary extension loaders so concurrent callers cannot observe temporary `sys.modules` entries before `exec_module()` completes.
- Preserve one-time shared-library loading, cached module identity, public loader APIs, and cleanup of failed initialization attempts without changing psychometric or numerical arithmetic.

#### Bradley-Terry control trust boundary

- `bradley_terry_mm()` now validates and normalizes `alpha`, `max_iter`, and `tol` before caller data materialization or compiled-core discovery. Exact built-in and supported NumPy scalar identities remain compatible; booleans, numeric subclasses, and arbitrary conversion providers are rejected without executing caller callbacks.
- `max_iter` is bounded by the package-wide `MAX_MAX_ITER` resource ceiling. Exact integers that overflow IEEE-754 conversion raise a package-owned `ValueError` before data materialization, matching the ICC adapter. Bradley-Terry MM arithmetic, convergence, normalization, estimates, and result statistics remain Rust-owned.

#### Delta-plot control trust boundary

- `delta_plot()` now establishes trusted selector, scalar, range, and iteration controls before materializing caller response/group data or discovering the compiled Rust core. Exact built-in strings and supported exact NumPy numeric scalar identities remain compatible; booleans, subclasses, unused-branch hostiles, and arbitrary conversion providers fail closed before caller callbacks. Huge exact integers that overflow `float()` raise package `ValueError` rather than a bare `OverflowError`.
- Normal-threshold `alpha` preserves the Rust `(0, 1)` domain, constraint ranges preserve `0 <= lo < hi <= 1`, fixed thresholds must be finite, additive adjustment counts stay positive, and `max_iter` is bounded by the package-wide `MAX_MAX_ITER` ceiling. Angoff Delta plot proportions, transforms, purification, thresholds, DIF flags, and result arithmetic remain Rust-owned.

#### Mantel-Haenszel control trust boundary

- `mantel_haenszel_dif()` now establishes trusted `fdr_q` and `exclude_studied_item` controls before materializing caller response/group data or discovering the compiled Rust core. Exact built-in bools and supported exact NumPy numeric scalar identities remain compatible; booleans-as-numbers, subclasses, and arbitrary conversion providers fail closed before caller callbacks. Huge exact integers that overflow `float()` raise package `ValueError` rather than a bare `OverflowError`.
- The FDR threshold preserves the existing finite `(0, 1]` domain. The ETS default still includes the studied item in the matching total. Mantel-Haenszel odds ratios, chi-square, ETS delta, standardized P-DIF, A/B/C classes, and BH flags remain Rust-owned.

#### Diagnostics-report focus and contrast preservation

- Revealed the visually hidden diagnostics-report skip link for every actual `:focus` state while retaining the explicit `:focus-visible` treatment and strong outline.
- Removed opacity-based dimming of non-hovered chart and table rows so unrelated active data retains its normal rendered foreground and background colors.
- Added public-renderer regression coverage and APA 7th doctoring grounded in WCAG 2.2 and Selectors Level 4 without making a formal conformance claim.

#### Git metadata lookup deadlines

- Commercial evidence builders fail closed with a bounded Git metadata timeout
  (benchmark, buyer packet, procurement, commercial release, Figma evidence)
  so hung `git rev-parse` cannot hang release pipelines.

#### Python support floor

- Raised `requires-python` to `>=3.12` so the advertised floor matches the
  hashed CI dependency lock (NumPy 2.5.x) and the pull-request matrix on
  CPython 3.12 and 3.14.

### Security

#### Bounded marginal latent-distance workspaces

- Rejected malformed dimensions, Boolean coercion, non-finite or non-float64
  matrices, hidden layout conversions, invalid epsilon, and oversized distance
  workloads with bounded non-reflective diagnostics.
- Added deterministic high-offset translation-stability, missing-data, and
  covariate parity tests plus a safe environment-reporting benchmark and APA 7
  operational doctoring.

#### Person-fit invalid-response error redaction

- Stopped reflecting caller-controlled invalid response values in `person_fit_np()` validation errors while preserving the failing matrix coordinate and the complete-data 0/1 response contract.

#### Bounded mixed item-model validation

- Bounded `item_models` iterable consumption to at most one look-ahead entry beyond the calibrated item count, rejected arbitrary non-string model controls before caller `__str__`/`__repr__` hooks can execute, and removed rejected model content from public validation errors while preserving accepted aliases and Rust-owned calibration numerics.

#### Model-comparison control validation hardening

- Reject hostile semantic/numeric control objects before caller-defined `__str__`, `__repr__`, or `__float__` callbacks can execute, while preserving accepted relation identities, built-in/NumPy scalar semantics, and Rust-owned model-comparison arithmetic.

#### Rotation mode validation hardening

- Reject non-built-in-string rotation `mode` controls before caller-defined representation callbacks can execute, while preserving the existing orthogonal/oblique vocabulary, aliases, default resolution, and Rust-owned rotation numerics.

#### ATA content-label validation trust boundary

- Keep invalid ATA content controls on a stable package-owned error surface rather than allowing arbitrary representation callbacks to execute during NumPy string coercion.

#### ATA semantic-range and exclusion preflight

- Keep ATA exclusion identities on an exact package-owned type/range boundary so Boolean, fractional, hostile integer-like, and out-of-bank values cannot be silently coerced or ignored before psychometric work.

#### G-theory public control validation

- G-theory D-study sizes and `Phi(lambda)` mastery cuts now reject unsupported caller objects before executing conversion or representation callbacks, while preserving supported Python/NumPy scalar controls and Rust-owned numerical behavior.

#### LSR ranking input bounds

- Bound LSR/I-LSR ranking CSR materialization (`MAX_RANKING_CSR_BYTES`, per-ranking `n+1` cap) and redact ordinary iterable failures at the Python validation boundary.

#### ATA constraint-map validation trust boundary

- Keep invalid ATA semantic controls on a stable package-owned error surface rather than allowing arbitrary `__str__`/`__int__`/`__index__` callbacks during constraint-map coercion.

#### Model-comparison callback-boundary hardening

- Harden parameter-count, audit-label, and real-valued model-comparison controls so caller-defined integer/string/NumPy subclasses and arbitrary integer-protocol providers are rejected before conversion or normalization callbacks execute, while preserving genuine NumPy scalar compatibility and Rust-owned Vuong arithmetic.

#### Bounded hourly PR queue capture

- Keep GitHub subprocesses bounded, retry only explicit HTTP 502/503/504 responses, reject malformed or duplicate PR identities, and fail rather than truncate queues above the supported cap.
- Enforce a 420-second cumulative monotonic capture deadline so sequential enrichment leaves time for deterministic failure manifests and artifact publication inside the ten-minute workflow job budget.

#### Cognitive-diagnosis native control boundary

- Validate CDM-family `max_iter`, `tol`, and DINA/DINO model selectors before compiled-core discovery. Only exact built-in values and explicitly supported concrete NumPy scalar types are normalized; booleans, subclasses, protocol providers, non-finite/out-of-range values, and unknown model selectors fail locally without executing caller conversion callbacks. Rust-owned psychometric arithmetic and result schemas are unchanged.

#### Answer-copying integer control boundary

- Harden Wollack omega, K-index, and K1/K2/S1/S2 row/count controls so only exact built-in integers and genuine supported NumPy integer scalars are normalized before compiled-core discovery; reject booleans, integer subclasses, and arbitrary coercion providers without executing caller callbacks.

#### Judge category-count RED completeness

- Extend `validate_judge(..., k=...)` regressions so `__index__`-only providers, comparison/repr hooks, booleans, `np.bool_`, 0-d arrays, and type-invalid controls fail before compiled-core discovery.
- Keep the existing trusted-scalar admission, `2..=1000` domain, and Rust-owned judge-validation arithmetic unchanged.
Closes #912.

#### Judge category-count control hardening

- Validate the public `validate_judge(..., k=...)` category count before compiled-core discovery.
- Accept exact built-in integers and genuine concrete NumPy integer scalars while rejecting booleans, subclasses, and arbitrary integer-conversion protocol providers without executing caller conversion callbacks.
- Marshal only a trusted built-in integer into the existing Rust-owned judge-validation computation; psychometric/fairness formulas, thresholds, and result schemas are unchanged.
Closes #912.

#### Harden Hofstee scalar control validation

- Harden Hofstee standard-setting scalar controls so rejected booleans, scalar subclasses, arbitrary conversion providers, non-finite/out-of-range percentages, overflowed trusted integers, and inverted bound pairs fail before Rust-core discovery; genuine supported NumPy scalars remain compatible and all Hofstee numerical arithmetic remains Rust-owned.

#### Harden many-facet control validation

- Harden public many-facet calibration controls so invalid booleans, scalar subclasses, arbitrary numeric protocol providers, unsupported quadrature/category/iteration controls, and non-finite or non-positive tolerances fail before Rust-core discovery; genuine supported NumPy scalars remain compatible and all MFRM numerical arithmetic remains Rust-owned.

#### Descriptor-safe bounded JSON input for automation scripts

- Consolidated governed automation JSON readers behind a descriptor-safe shared
  loader with a 32 MiB inclusive byte bound and a non-recursive 128-level depth
  bound.
- Rejected symbolic links, FIFOs, directories, path replacement, invalid UTF-8,
  malformed JSON, non-object roots, oversized input, and excessive nesting with
  deterministic tests.

#### Bounded JSON loads for release acceptance

- Release acceptance and generation-request contract loading use size- and
  depth-bounded JSON parsers instead of unbounded `json.loads` on CLI stdout
  and fit_summary artifacts.

#### Reliability integer control boundary

- Harden Finn `s_levels`, Guttman `n_sample_splits`/`seed`, and Feldt `n_persons`/`n_items` so only exact built-in integers and genuine supported NumPy integer scalars are normalized before compiled-core discovery or ratings materialization; reject booleans, integer subclasses, and arbitrary coercion providers without executing caller callbacks.

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
  release notes) now form the `[0.5.0] - 2026-08-03` release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

## [0.5.0] - 2026-08-03

### Added

#### Pilot-observation handoff to observed-score DIF screening

- A factory-sealed, content-addressed `DifPilotDesign` assembled by
  `build_dif_pilot_design` from replay-verified pilot observation records.
  It wraps the binary MIRT assembler's fail-closed response, provenance,
  missingness, duplicate-cell, category, observed-support, and
  dense-allocation contracts, and requires descriptive reference/focal
  group identifiers with an exact assignment for every indexed respondent;
  missing, unknown, undeclared, one-group, and normalized-collision group
  contracts are rejected with structured error codes.
- `to_observed_score_dif_kwargs` emits copied `responses` and `group`
  arrays accepted directly by the repository's Rust-backed binary
  observed-score DIF APIs (`mantel_haenszel_dif`, `logistic_dif`, their
  purified variants, and `sibtest`) and refuses incomplete matrices
  instead of silently applying complete-case deletion or imputation. The
  handoff performs no psychometric arithmetic and makes no invariance,
  fairness, or validity claim. The G-theory handoff remains the final
  follow-up slice of issue #407.

#### Pilot-observation handoff to one-facet G theory

- A factory-sealed, content-addressed `GTheoryPiPilotDesign` assembled from the
  existing replay-verified many-facet pilot design. It preserves complete item
  and response provenance while requiring exactly one rater, one declared
  occasion, at least two respondents, at least two items, and an explicitly
  observed score in every respondent-item cell.
- `to_gtheory_pi_kwargs` and `to_phi_lambda_kwargs` emit copied `float64`
  persons-by-items matrices plus bounded D-study item counts and a finite mastery
  cut accepted directly by the repository's Rust-backed one-facet G-theory APIs.
  Missingness, case deletion, imputation, rater aggregation, and score coercion
  fail closed rather than being hidden at the handoff boundary.
- The artifact is explicitly limited to the complete balanced `p x i` design.
  It does not relabel raters as occasions or fabricate a `p x i x o` tensor from
  item-level occasion provenance; a multi-occasion bridge remains deferred until
  the schema can bind repeated administrations to a stable cross-occasion item
  family. No universal coefficient cutoff, variance-component policy, fairness,
  scoreability, or validity claim is made.

#### Generated-item pilot handoff to testlet calibration

- A factory-sealed, content-addressed `TestletPilotDesign` assembled from the
  existing replay-verified binary pilot design. It preserves exact missingness,
  per-cell rater provenance, item provenance, and governed
  `query_testlet_id` groupings while emitting copied `responses` and
  integer `testlet_id` arrays accepted by the Rust-backed `fit_testlet` API.
- Singleton-only groupings are rejected instead of being labelled as a
  testlet design. Rasch/2PL selection, iteration limits, quadrature size,
  tolerance, variance initialization, and convergence policy are validated
  before calibration arguments are returned.
- The handoff performs no psychometric arithmetic and makes no connectedness,
  convergence, local-dependence, fit, reliability, fairness, scoreability, or
  validity claim. Model comparison, parameter recovery, residual diagnostics,
  DIF/fairness analysis, and human-anchored validity evidence remain required
  before operational use.

### Changed

#### Release cut 0.4.0

- Project version is bumped to 0.4.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the binary MIRT and bifactor pilot-calibration
  handoffs for generated-item pilot observations, the vectorized NumPy
  MMLE fallback M-step, and the semantic hero-metadata report
  accessibility improvements) now form the `[0.4.0] - 2026-08-03`
  release section.
- Released authoritative fragments are removed from `docs/changelog.d`;
  the directory again holds only genuinely unreleased notes.

### Fixed

#### Changelog render-parity gate and 0.4.0 note restoration

- The `[0.4.0]` CHANGELOG section again carries the binary bifactor
  pilot-calibration handoff, the vectorized NumPy MMLE fallback M-step, and
  the accessible hero-metadata notes. Their authoritative fragments had been
  merged without re-rendering the managed Unreleased block, so the 0.4.0
  release cut inherited a stale block and the published immutable v0.4.0
  GitHub release body silently omitted them; `CHANGELOG.md` is the
  authoritative release record for v0.4.0.
- A repository-level regression test now runs the fragment renderer's check
  against the live `CHANGELOG.md` and `docs/changelog.d` tree, so a pull
  request that adds or edits a fragment without rendering it fails CI before
  any release cut can inherit a stale managed block.

## [0.4.0] - 2026-08-03

### Added

#### Pilot-observation handoff to binary MIRT calibration

- A factory-sealed, content-addressed `MirtPilotDesign` assembled by
  `build_mirt_pilot_design` from replay-verified pilot observation records:
  a deterministic persons-by-items binary response matrix with explicit
  `missing`, `not_applicable`, and `insufficient_evidence` states preserved
  alongside the numeric `NaN` representation, per-cell rater assignments
  retained as provenance, and copied `responses`/`factor_id` arguments
  accepted directly by the existing Rust-backed `fast_mlsirm.fit` API.
- Items are assigned to trait dimensions by sorted `query_testlet_id`
  (simple-structure: one dimension per query testlet), and the full mapping
  is disclosed through `factor_testlet_ids`/`item_factor_ids` inside the
  content-addressed design identity.
- The handoff is fail-closed: mixed pilot studies, conflicting item
  provenance, duplicate respondent-item cells (multi-rater data must use
  `build_facets_pilot_design`), non-binary observed categories (polytomous
  data likewise), unobserved respondents or items, and dense designs above
  the documented `MAX_MIRT_PILOT_CELLS` budget are all rejected with
  structured `PilotObservationError` codes before allocation. No silent
  rater aggregation or dichotomization is ever performed, and the handoff
  makes no scoreability, fit, or validity claim. BIFAC2PLM, testlet, DIF,
  and G-theory handoffs remain follow-up slices of issue #407.

#### Pilot-observation handoff to binary bifactor calibration

- A factory-sealed, content-addressed `BifactorPilotDesign` assembled by
  `build_bifactor_pilot_design` from the existing replay-verified binary pilot
  design. The artifact records one descriptive general-factor identity over
  every item, preserves each item's governed `query_testlet_id` as its sole
  specific-factor assignment, retains exact missingness and rater provenance,
  and emits copied `responses`, `factor_id`, and `FitConfig` arguments accepted
  directly by the Rust-backed `fast_mlsirm.fit` API.
- The handoff pins `model="BIFAC2PLM"`, `estimator="mmle"`, and
  `latent_dim=1`; caller-tuned numerical settings are accepted only when those
  structural constraints remain intact. It reuses the binary MIRT assembler's
  fail-closed provenance, duplicate-cell, category, observed-support, and dense
  allocation contracts rather than introducing a weaker parallel parser.
- Buyer documentation states the classical all-items general-factor plus
  at-most-one-specific-factor pattern and the downstream identification,
  model-comparison, scoreability, DIF/fairness, recovery, and validity gates.
  A successful handoff is not a fit, scoreability, fairness, or deployment
  claim. Testlet, DIF, and G-theory handoffs remain follow-up slices of issue
  #407.

### Changed

#### Accessible diagnostics-report hero metadata

- Generated HTML diagnostics reports now expose the source filename as a
  semantic description list and hide redundant decorative branding from the
  accessibility tree.
- Regression coverage pins the `aria-hidden`, `dl`, `dt`, and `dd` markup so
  future template changes cannot silently remove the accessibility contract.

#### Vectorized NumPy MMLE fallback M-step

- The unidimensional 2PL NumPy fallback now updates active item parameters with
  vectorized Newton operations instead of a per-item Python loop. The compiled
  Rust MMLE path remains the preferred production implementation.
- A deterministic missing-data regression fixture preserves discrimination,
  intercept, EAP, and log-likelihood results within explicit floating-point
  tolerances; no bitwise-equivalence or environment-independent speedup claim
  is made.

#### Release cut 0.3.0

- Project version is bumped to 0.3.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (the fail-closed manual release-tag workflow with body
  capping and interrupted-run resume, the accessible exact-value report
  disclosure, the precomputed marginal distance hot paths with
  context-sharded threading, the currency-explicit enterprise due-diligence
  gate, duplicate-issue-claim PR governance, the generated-item audit and
  pilot admission gate with its facets observation handoff and resource
  bounds) now form the `[0.3.0] - 2026-08-03` release section.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.

## [0.3.0] - 2026-08-03

### Added

#### Generated-item audit and pilot-admission gate

- A deterministic, content-addressed `CandidateAuditReport` that inspects only
  parser-validated generated-item candidates and emits bounded redacted
  findings for instruction-override indicators, ambiguity-prone option or
  stem patterns, duplicate normalized option/evidence surfaces, overlapping
  rubric indicators, non-atomic criterion indicators, normalized duplicate
  source attributions, declared safety notes, and excessive finding volume.
- An enforced `draft -> audited -> pilot` lifecycle: blocking or
  review-required findings retain the candidate in `draft`; only an exact
  candidate/audit fingerprint match may produce an immutable
  `PilotCandidateRecord` with explicit pilot-study, query/testlet, generator,
  judge-policy, occasion, rubric, blueprint, and item provenance.
- Public pilot admission replays the package's exact named and versioned audit
  policy and compares the complete report fingerprint, preventing a caller
  from constructing a clean-looking report that bypasses real findings or
  relabeling the current implementation as an unsupported policy version.
- The top-level public `PilotCandidateRecord` is factory-sealed against
  ordinary direct construction. This is an API-governance boundary rather
  than a cryptographic capability; downstream systems must verify the complete
  candidate, audit-report, and pilot-record fingerprints.
- The audit is a deterministic screening and governance boundary, not a
  semantic answerability, fairness, scoreability, psychometric validity, or
  operational-deployment declaration. Pilot observation conversion and
  Rust-backed calibration remain follow-up slices of issue #407.

#### Pilot-observation handoff to many-facet calibration

- Factory-built, content-addressed `PilotObservationRecord` values that bind
  respondent and rater responses to the complete replay-verified pilot item
  provenance.
- Explicit `observed`, `missing`, `not_applicable`, and
  `insufficient_evidence` states; non-observed states cannot carry a category
  and are never coerced to failure scores.
- A deterministic `FacetsPilotDesign` assembler that rejects mixed studies,
  conflicting item provenance, duplicate cells, invalid category regimes, and
  unobserved indexed facets before producing copied arguments for the existing
  Rust-backed `fit_facets` API.
- Content-addressed preservation of the exact response-state tensor alongside
  the numeric `NaN` representation used for many-facet estimation.
- This handoff performs no psychometric arithmetic and makes no adequacy,
  connectedness, fairness, scoreability, calibration, or validity claim.
  MIRT, bifactor, testlet, DIF, and G-theory handoffs remain follow-up slices
  of issue #407.

#### Accessible exact-value disclosure in HTML reports

- A shared `report_exact_values` component that renders every plotted report
  section's complete, untruncated source rows as an open-by-default native
  `details`/`summary` disclosure with a semantic full-precision table
  (float cells use shortest round-trip `repr`, missing cells are named
  explicitly) plus copyable JSON and CSV exports generated from the same
  rows (strict JSON with `null` missingness; RFC 4180 CSV).
- Fit and dimensionality report sections now carry the disclosure adjacent
  to their charts, so exact values stay available on touch input, in
  print/PDF output, under keyboard-only navigation, and with JavaScript
  disabled, per WCAG 2.2 success criteria 1.3.1, 1.4.13, 2.1.1, and 4.1.2
  (issue #409). Chart markup and the summarized 12-row tables are unchanged.

#### Fail-closed duplicate issue-claim governance

- PR queue governance now parses `Closes`, `Fixes`, and `Resolves` references
  from every active pull-request body and blocks the commercial release gate
  when multiple open PRs claim one issue without exactly one issue-specific
  `Canonical-For: #N` designation.
- Deterministic JSON and accessible standalone HTML evidence now include the
  repository base SHA, active PR head SHAs, issue references, timestamps,
  duplicate-head warnings, high changed-file-overlap warnings, and bounded
  closed/merged claim history.
- One-file intersections are excluded from changed-file duplicate warnings, and
  the hourly read-only workflow uploads governance artifacts even when the
  duplicate-claim gate fails.

#### Currency-explicit enterprise due-diligence gate

- Added a deterministic `enterprise_due_diligence_gate` manifest utility that separates the amount-neutral software evidence gate from a currency-explicit procurement scenario and always records `valuation_claim: false`.
- Added a bounded deprecation bridge for legacy `20b` gate aliases and `--require-20b-product`, with canonical output and explicit warnings.
- Documented the distinction among enterprise evidence, the KRW 2,000,000,000 procurement scenario, and the aspirational USD 20,000,000,000 enterprise-value thesis.

#### Manual release-tag workflow

- A fail-closed `workflow_dispatch` release workflow (`release-tag.yml`) that
  verifies the requested semantic version matches `pyproject.toml` and an
  existing CHANGELOG release section, extracts that section as the release
  notes, refuses to overwrite an existing release, and publishes the `v*` tag
  and GitHub Release with a least-privilege job-scoped `contents: write`
  token. Because a `GITHUB_TOKEN`-created tag does not retrigger tag-push
  workflows, the exhaustive Statistical Studies evidence run is dispatched
  separately after a release.
- A release body larger than GitHub's 125,000-character release limit is
  replaced by a bounded summary that links the authoritative CHANGELOG
  section and lists the released feature headings.
- A run interrupted between tag creation and release publication resumes on
  the existing immutable tag: an existing release always blocks, an existing
  tag without a release skips re-tagging and publishes against the verified
  tag, and API uncertainty still fails closed.

### Changed

#### Precomputed marginal distance hot paths

- The marginal estimator's latent interaction term
  `sqrt(eps_distance + ||x - zeta_i||^2)` depends only on the
  `(item, latent-node)` pair, so the probability-table build, the M-step item
  line search, the item gradient, and the tau gradient now reuse a
  precomputed `n_items x n_x` table instead of recomputing the distance and
  `exp(tau)` once per `(context, item, trait-node, latent-node)` quadrature
  cell. Distance-kind results are bit-identical to the scalar reference by
  construction (`a - b == a + (-b)` in IEEE-754, identical accumulation
  order); inner-product models agree to documented floating-point round-off.
- The marginal probability-table fill is sharded over population contexts
  with the crate's coarse fixed-shard `thread::scope` pattern above a
  documented cell floor, with a test-forcible worker seam proving the sharded
  fill bit-identical to the serial fill.
- A same-head benchmark contract
  (`marginal_distance_benchmark_reports_runtime_and_allocation`) emits JSON
  evidence with runtime, output-table and interaction-table bytes, workload
  dimensions, backend label, and the documented 1.05x regression threshold.
  Representative release-mode workload (6 contexts, 60 items, 21 trait
  nodes, 121 latent nodes): scalar reference 37.9 ms, precomputed serial
  20.8 ms, four-worker shard 8.0 ms, interaction-table overhead 58 KB
  against 14.9 MB of output tables (issue #403). No Python estimator or
  fallback arithmetic changed.

#### Release cut 0.2.0

- Project version is bumped to 0.2.0 in `pyproject.toml`,
  `crates/mlsirm-core`, and `crates/fast-mlsirm-py`, and the accumulated
  `Unreleased` notes (bifactor scoreability indices, rubric blueprint
  compiler, provider-generated item validation, adaptive exploratory factor
  rotation and criterion selection, the Rust-only literature true-parameter
  recovery gate, fail-closed Vuong selection summary, GPU-preferred serving
  and information kernels, and CI queue governance) now form the
  `[0.2.0] - 2026-08-03` release section.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.

### Fixed

#### Facets-pilot resource bounds

- Reject persons-by-items-by-raters dense pilot designs above 1,000,000 cells
  before allocating Python tuple tensors or NumPy arrays, preventing bounded
  sparse observations from amplifying into memory-exhaustion workloads.
- Validate observed respondent, item, and rater support with precomputed sets
  before dense construction, replacing repeated full scans with linear-time
  membership checks.

#### Factor-CSV supplementary-plane input guard

- `load_factor_csv` now rejects CSV content containing characters outside the
  Unicode Basic Multilingual Plane with a benign `ValueError` before the text
  reaches `numpy.loadtxt`. The coverage-guided fuzz harness found that NumPy's
  loadtxt tokenizer crashes the process on some supplementary-plane characters
  (for example U+109249) and silently converts others (for example U+10000)
  into garbage integer factor ids; the crashing input is preserved as a fuzz
  corpus regression seed alongside pytest reproducers.

## [0.2.0] - 2026-08-03

### Changed (serving and information kernels)

- Fixed-bank item/test information and CAT information selection now default
  to a Rust wgpu kernel, retain an explicit Rust f64 CPU path, and accept the
  same `device="auto"|"gpu"|"cpu"` contract as EAP serving. Non-finite f32
  results are discarded in favor of the finite CPU reference.
- Plausible-value posterior reduction and seeded sampling now use the same
  GPU-preferred device contract. Unsupported GPU sizes or results fall back to
  a deterministic Rust f64 implementation with fixed contiguous CPU shards.
- EAPsum table recursion, posterior moments, and respondent lookup now remain
  in Rust and prefer wgpu. Explicit CPU execution retains the f64 reference;
  unavailable or unsupported GPU work falls back to fixed contiguous Rust CPU
  workers instead of performing respondent score aggregation in Python.

### Changed

- Added `prefers-reduced-motion` and print media-query support to every
- Rust EAP scoring now defaults to GPU-preferred `auto` execution in the core,
  PyO3 binding, and serving API. The f64 CPU reduction remains available via
  `device="cpu"`; an explicit unavailable `device="gpu"` request now warns
  before falling back instead of silently running on the CPU. Serving EAP now
  requires the compiled Rust core instead of silently bypassing device policy
  through the NumPy reference implementation.

### Security

- **Input-validation hardening at the untrusted boundaries** (Strix scan
  findings on PR #160). All are denial-of-service / data-poisoning guards for
  a library that may be exposed as a scoring/fitting service:
  - Population labels (`group_id`/`cluster_id`) are now validated and
    **compacted to contiguous ids** in `fit.py` and `inference.py`, so the
    group/cluster count is the number of *distinct* labels (≤ `n_persons`)
    rather than `max(label)+1` — sparse ids like `[0, 1e9]` no longer force
    billion-row population allocations. Negative, non-integer, non-finite, and
    wrong-length labels are rejected.
  - `FitConfig.validate()` bounds `latent_dim` (≤ `MAX_LATENT_DIM = 8`),
    `xi_points` (≤ `1_000_000`), `max_iter` (≤ `100_000`), `n_restarts`
    (≤ `1_000`), and `m_steps` (≤ `1_000`), and rejects **non-finite**
    `learning_rate`/`init_gamma`/`eps_distance`/`tolerance`/`gradient_clip`
    (a bare `x <= 0` comparison lets `NaN`/`Inf` through) — blocking both
    memory/CPU exhaustion from extreme sizes and NaN-poisoned fits.
  - `plausible_values` bounds `n_draws` (1..`MAX_DRAWS = 100_000`), and
    `serving_prior` bounds `n_dims` (1..64) for direct callers.
  - `load_serving_bundle` parses JSON in **strict mode** (rejects `NaN`/
    `Infinity` literals) and runs a full `_validate_bundle` structural +
    finiteness check (consistent `n_items`/`n_dims`/`latent_dim`, bounded
    sizes, in-range `factor_id`, finite `alpha`/`b`/`zeta`/`tau`/`eps_distance`,
    supported quadrature); `score_respondents` and `plausible_values` validate
    the bundle at entry, so oversized dimensions (e.g. `n_items = 1e12`) and
    non-finite parameters can no longer trigger multi-terabyte allocations or
    NaN scores.
  - `plausible_values` now enforces the binary response domain (0/1, finite)
    that `score_respondents` already required.
  - `validate_judge` validates judge/human/baseline/subgroup labels (1-D,
    equal length, finite, integer, `0 ≤ label < k`) **before** the `uint32`
    conversion, instead of silently truncating floats or wrapping negatives.
  - Regression tests in `tests/test_security_hardening.py` cover each finding.
- **Second-pass hardening** (Strix re-scan of PR #160, 11 findings) extends the
  same DoS/data-poisoning guards to the paper-feature surface added in this PR:
  - `preprocessing.irtree_expand` bounds the dense expansion
    (`persons * items * nodes ≤ 50_000_000`) before allocating, and validates
    `node_dims` (finite, non-negative, integer-valued) before the `int64` cast.
  - `validation._validate_labels` rejects labels above `uint32` max before the
    narrowing cast, and `validate_judge` requires the `human_human` baseline to
    match the paired sample size.
  - `inference.observed_information` caps the finite-difference Hessian at
    `5_000` parameters (it is `O(n²)` memory **and** `O(n²)` objective calls),
    and `oakes_standard_errors` validates `factor_id` (1-D, one-per-item,
    finite, non-negative, integer) before deriving `n_dims`.
  - `serving._validate_bundle` rejects tensor Gauss-Hermite grids that would
    allocate `q_xi ** latent_dim > 1_000_000` points; `estimators.marginal`'s
    `_xi_grid` carries the same bound for direct callers.
  - `linking.link_fixed_item_parameters` rejects duplicate/fractional/negative/
    non-finite anchor indices, non-2-D `theta`, non-finite item parameters, and
    non-finite computed linking coefficients.
- **Third-pass hardening** (Strix re-scan of `b5d9d90`, 11 real findings; the
  12th — "incomplete package release" — was a scanner artifact of its
  PR-scope-only checkout, verified: every named module exists and
  `import fast_mlsirm` succeeds) **plus a proactive boundary audit** that found
  6 more Python issues Strix had not surfaced:
  - `serving.score_respondents`/`plausible_values` bound the dense respondent
    matrix (`rows x n_items`); `serving._validate_bundle` now bounds the
    scoring-table product (`max(n_items, n_dims) x q_theta x q_xi**latent_dim` —
    a 55+ GB allocation otherwise) and validates the bundle `population` block
    (`serving_prior` computed `sqrt(1 + sigma_u**2)` on an unvalidated, fully
    attacker-controlled `sigma_u` → `TypeError`/`OverflowError` crash or silent
    `Inf`/`NaN` score poisoning).
  - `linking.link_fixed_item_parameters` range-checks anchor indices on the
    float **before** the `int64` cast (`uint64` max silently wrapped to `-1`,
    selecting the last item) and requires a positive linking scale;
    `linking.irt_link` validates slope/intercept finiteness and slope
    positivity before the Nelder-Mead core (a `NaN` would panic it).
  - `validation.validate_judge` bounds the category count `k` (drives a dense
    `k x k` confusion matrix) and **compacts** sparse `subgroup` labels (the
    core loops `0..max(label)+1`, an O(4e9) CPU-DoS from one sparse id).
  - `preprocessing.irtree_expand` switched from a 50M-element ceiling (400 MB,
    boundary-inclusive) to a 64 MiB byte budget; `config.MLS2PLMConfig.validate`
    bounds simulation dimensions and the `n_persons x n_items` cell product;
    `config.FitConfig.validate` bounds aggregate optimizer work
    (`max_iter x n_restarts`); `estimators.marginal.fit_marginal_numpy` bounds
    declared population counts (`n_groups`/`n_clusters <= n_persons`) and the
    EM working set; `inference.observed_information` rejects non-finite `step`;
    `inference.oakes_standard_errors` and every `fitstats` public entry bound
    `n_dims` derived from an untrusted `factor_id` (a shared `_validate_factor_id`
    guard); `fit.py` validates anchor/covariate array shapes and finiteness
    before the Rust marginal core.
  - Rust-core backstops for the same audit (defense in depth, active once the
    extension is rebuilt): `fitstats::s_x2` rejects non-dichotomous observed
    responses (a non-0/1 value indexed the summed-score table out of bounds →
    panic); `fitstats::infit_outfit` validates `theta`/`xi` lengths before
    indexing; `scoring::validate_prior` rejects non-finite prior `mean`/`sd`
    (a `NaN` `sd` passed the bare `sd <= 0` check).
- `factor::validate_corr` now rejects off-diagonal correlations outside
  `[-1, 1]` (impl-review finding): an impossible value like `1e308` passed
  the old finiteness/symmetry checks and panicked inside the eigen sort
  instead of returning an error (affected `minres_fa`, `omega_total_1f`,
  and the new `glb_fa`); regression-tested.

### Added

- Sample size for Cohen's kappa (`n_cohen_kappa`, Rust core
  `mlsirm_core::reliability::n_cohen_kappa` + thin Python wrapper):
  closed-form required-subject count for a one- or two-sided test of
  H0: kappa = k0 vs H1: kappa = k1 on a 2x2 two-rater table, transcribed
  from the CRAN irr 0.84.1 R source `N.cohen.kappa.R` (read in full; irr
  attributes the method to Cantor, 1996, NOT read, cited as origin only).
  Returns `n`, variance factors `q1`/`q0`, and the pre-ceiling size.
  Stricter than R: degenerate marginals, infeasible implied cell
  probabilities, and nonpositive variance factors raise errors where R
  silently returns NaN. Evidence: exact-Fraction oracle executed
  (4 pinned fixtures + infeasibility probe; Acklam-vs-NormalDist ceil
  stability asserted); adversarial spec review APPROVED-WITH-CHANGES,
  all 4 mandatory changes adopted (rate-swap symmetry disclosed as a
  real unkillable identity); 6/6 mutation kills EXECUTED (pie term,
  Q-role swap, sidedness, pi22 halving, floor-for-ceil, unsquared
  denominator); MC-500 secondary-oracle sweep passes.
- Rater bias chi-square `rater_bias` (Rust core
  `mlsirm_core::reliability::rater_bias` + thin Python wrapper), transcribed
  from the CRAN irr 0.84.1 R source `rater.bias.R` (read in full; the
  statistic is McNemar-style but McNemar, 1947, was NOT read and is not
  cited as normative). With `rbb`/`rbc` the strict upper/lower triangle
  sums of a CxC two-rater table: `value = rbb/(rbb+rbc)`,
  `statistic = (rbb-rbc)^2/(rbb+rbc)`, df = 1, upper-tail chi-square p;
  `subjects` sums ALL cells (R: `sum(rbx)`). REDUCED-SCOPE vs R:
  CxC-table branch only (the nx2/2xn raw-score `table()` front-end is a
  plain cross-tab left to callers, as for `stuart_maxwell_mh`). Per the
  adversarial spec review, the `2^53/(2C)` per-cell cap does NOT make f64
  triangle sums exact for large C, so `rbb`/`rbc` accumulate in u64
  (exact, bounded by 2^61) and the squared difference is formed in i128;
  f64 rounding only in the final divisions. Deliberately stricter than R:
  explicit errors for non-square input, <2 or >1000 categories,
  negative/NaN/Inf/non-integral counts, cells above the cap, and
  `rbb + rbc == 0` (R would form 0/0). Disclosed unkillable mutant:
  removing R's `abs()` (no-op on the validated nonnegative domain).
  Exact-Fraction oracle anchors (3x3 value 10/13 stat 147/13; 4x4 on the
  value<1/2 side 4/9, 1/3; dyadic 2x2 3/4, 2 asserted exactly; balanced
  rbb==rbc stat 0 p 1; diagonal-only error), a transpose-antisymmetry
  test (stat/p/n invariant, values sum to 1), 5/5 targeted mutants
  executed-killed (triangle swap, difference denominator, off-diagonal
  subjects, diagonal in rbb, df 2), and an in-repo `#[ignore]` MC-500
  independent-recompute test.
- Bhapkar marginal homogeneity test `bhapkar_mh` (Rust core
  `mlsirm_core::reliability::bhapkar_mh` + thin Python wrapper), transcribed
  from the CRAN irr 0.84.1 R source `bhapkar.r` (read in full; Bhapkar, 1966,
  cited as origin only, not read). Statistic `d' W^-1 d` with
  `W = S - d d'/n` over the first C-1 categories (no equal-marginal drop,
  unlike Stuart-Maxwell), df = C-1, upper-tail chi-square p. Verified against
  an executed exact-Fraction oracle (pins: 3x3 stat 196080/18733, 4x4 with a
  kept equal marginal, 2x2 15/7, perfect-agreement singular, zero-d table);
  the oracle also proves exactly the identity `bhapkar = SM/(1-SM/n)` against
  the no-drop Stuart-Maxwell statistic (cross-implementation test anchor) and
  drop-invariance. 5/5 targeted mutants executed-killed (W sign, missing /n,
  skipped dd'/n correction, df, S transpose cell); an in-repo `#[ignore]`
  MC-500 test cross-checks a local independent solve plus permutation
  invariance. Singularity uses the scaled pivot threshold
  `1e-12 * max|W|` from the start.
- `stuart_maxwell_mh` Stuart-Maxwell marginal homogeneity chi-square test for a C×C two-rater counts table (CRAN irr 0.84.1 `stuart.maxwell.mh()` `R/stuart.maxwell.R` — READ and normative; Stuart 1955 and Maxwell 1970 NOT READ, cited as method origin only): one-shot simultaneous drop of every category with equal row/column marginals (R does NOT re-check equality after the drop — preserved verbatim), then on the remaining K categories `d_i = r_i − c_i`, `S_ii = r_i + c_i − 2x_ii`, `S_ij = −(x_ij + x_ji)` over the first K−1, statistic `d'S⁻¹d` with df = K−1 and upper-tail chi-square p (crate `chi2_sf`, a regularized-upper-incomplete-gamma transcription whose absolute error at the pinned p-values is ≲6e-16 per an executed check). REDUCED-SCOPE vs R: counts-table branch only (R's n×2 raw-score branch is a plain cross-tab left to callers). Deliberately stricter than R: explicit errors for non-square input, <2 categories before/after the drop, negative/NaN/Inf/non-integral counts, cells above 2^53/(2C) (exact f64 marginal sums; subjects accumulated in checked u64), >1000 categories, and singular S (checked Gaussian elimination per the `lltm::solve_small_checked` pattern — no silent fallback). Disclosed unkillable identities: d→−d (quadratic form) and solver transposition (S symmetric by construction). Exact-Fraction oracle anchors M1–M6 (3×3 stat exactly 1520/157 with p 0.007901012752471986; 4×4 drop-path stat 200/171 pinned equal to the crate's direct-submatrix run; 2×2 McNemar reduction (6−2)²/8 = 2 exactly; permutation invariance; all-equal-marginals error), a 6-mutant EXECUTED kill map via a source-mutation harness (S_ii sign, S_ij sign, d = r+c, df = K, skipped drop, missing transpose cell), and an in-repo `#[ignore]` MC-500 test with an independent Cramer-rule recompute plus random simultaneous-permutation invariance.
- `mean_pairwise_rho` mean of the pairwise Spearman rank correlations between rater columns (CRAN irr 0.84.1 `meanrho()` `R/meanrho.R` — READ and normative; Spearman 1904 NOT READ, cited as method origin only): listwise NaN row drop, midrank transform per column (R `rank` default, tie groups averaged; the hand-derived Spearman = Pearson-on-midranks equivalence is verified by an executed exact-Fraction oracle), then the `mean_pairwise_cor` machinery on ranks — plain mean (`fisher=False`, statistic/p None) or Fisher-z average `tanh(mean(atanh ρ))` with strict ±1 pair exclusion (`dropped` reported), `SE = √(1/(m−3))`, two-sided `p = erfc(|z|/√2)`. A `ties` flag reports duplicates within any column; a documented case analysis shows R's `apply(..., unique)` matrix-collapse quirk never diverges from this flag for nr ≥ 2. Deliberately stricter than R: constant rank columns, fewer than 4 complete rows under fisher, and all-perfect pair sets are explicit errors. Exact-Fraction oracle anchors S1–S6 (reversed pair cancels z's to value 0, cyclic equal-rho fixture kills the missing-tanh mutant exactly at 23/35, ties fixture pins the irrational midrank rho 0.8922178162191939, NaN-row fixture is bitwise-equal to its complete counterpart, distinct-rho fixture 0.6992488340329346 discriminates Fisher averaging, duplicate-column fixture pins dropped=1 at 0.7447132997063424); a 7-mutant kill map (raw values, ordinal ranks, missing tanh, SE m−1, perfect pairs kept, listwise drop skipped, zero-on-drop) executed against the in-repo Rust tests via a source-mutation harness; and an in-repo `#[ignore]` MC-500 test with an independent midrank+Fisher recompute and column-swap invariance.
- `mean_pairwise_cor` mean of the pairwise Pearson correlations between rater columns (CRAN irr 0.84.1 `meancor()` `R/meancor.R` — READ and normative; Fisher 1925 NOT READ, cited as the z-transformation origin only): listwise NaN row drop, all `C(nr,2)` pairwise correlations with a single-square-root denominator so exactly (anti)proportional columns give r = ±1 exactly, and either the plain mean (`fisher=False`, perfect pairs kept, statistic/p None) or the Fisher-z average `tanh(mean(atanh r))` with strict ±1 pair exclusion (`dropped` reported where R only warns), `SE = √(1/(m−3))`, and two-sided `p = erfc(|z|/√2)` clamped to [0, 1]. Deliberately stricter than R: constant columns, fewer than 4 complete rows under fisher, and all-perfect pair sets are explicit errors. Exact-Fraction oracle anchors C1/C2/C7/C8 (permutation columns give exactly rational r: fisher value 0.5350920914541507 at m=4, m=7 statistic = 2·value pinning the SE division past the m=4 SE=1 identity, r=−1 boundary drop, plain means 7/15, 19/30, 17/21, −1/6), a 6-mutant EXECUTED kill map (tanh omitted, filter dropped, SE m−1, one-sided p, skipped pair, skipped listwise drop), and an MC-500 independent-recompute + column-reversal invariance test.
- Robinson's A coefficient of agreement `robinson_a` (Rust core + Python wrapper), transcribed from the CRAN irr 0.84.1 `robinson.R` source with listwise NaN deletion; degenerate inputs with no subject variance raise errors where R silently returns NaN.
- `maxwell_re` Maxwell's RE agreement coefficient for two raters with binary ratings (CRAN irr 0.84.1 `maxwell()` `R/maxwell.R` — READ and normative; Maxwell 1977 NOT READ, cited as method origin only): `RE = 2*A/ns - 1` where `A` counts exact-match subjects after listwise NaN deletion; the distinct-value union across both columns must have at most 2 levels (any two numeric labels accepted; single-level input yields 1). The R diagonal-of-`table(r1, r2)` form is hand-derived to equal the match count regardless of the level-ordering quirk and verified against an executed exact-Fraction oracle. Deliberately stricter than R: explicit errors for `nr != 2`, infinities, and empty input. Rust core with a thin Python wrapper.
- `finn_coefficient` Finn (1970) reliability coefficient for discrete-scale ratings (CRAN irr 0.85 `finn()` `R/finn.R` — READ and normative; Finn 1970 NOT READ, cited as method origin only): compares the within-subject (oneway `MSw`, mean of per-row sample variances) or two-way residual (`MSe`) mean square against the discrete-uniform expectation `MSexp = (s²−1)/12`, `coeff = 1 − MS/MSexp`, `F = MSexp/MS`, with the R quirk that BOTH models use `df2 = ns(nr−1)` preserved verbatim. The upper-tail p-value implements `pf(F, Inf, df2, lower.tail=FALSE)` via the hand-derived limiting identity `P(F > f) = P(χ²_df2 < df2/f)` (convergence-verified against scipy; valid for F > 0 — negative mean squares from floating cancellation are explicit errors, and perfect agreement returns value 1, statistic +Inf, p 0). Rows with NaN are dropped listwise (R `na.omit`); infinities, `s_levels < 2` (or bool), and fewer than 2 complete rows/raters are explicit stricter-than-R errors. Exact-Fraction oracle anchors FN1–FN4 (oneway 125/144 with p 4.47127350746514e-06, twoway 617/720, listwise-drop 13/15, negative twoway coefficient −1/8), a 6-mutant EXECUTED kill map (MSexp off-by-one, population variance, dropped MSc term, wrong df2, flipped p tail, skipped listwise drop), and an MC-500 subject/rater permutation-invariance test.
- `light_kappa` Light's kappa for multi-rater nominal agreement (CRAN irr 0.85 `kappam.light()` `R/kappam.light.R` + unweighted branch of `kappa2()` `R/kappa2.R` — both READ and normative; Light 1971 NOT READ, cited as method origin only): mean of the `C(nr,2)` pairwise unweighted Cohen's kappas after listwise missing-row drop, plus Light's chance-product z test `disraterₚ = m² − Σₐc1[a]c2[a]`, `chanceP = 1 − npairs·Π(disraterₚ/m²)` (overflow-safe form, algebraically identical to R's `1 − B/m^(2·npairs)`), `varκ = chanceP/(m(1−chanceP))`, `z = value/√varκ`. R builds each pair's level set from the two columns only; this implementation compacts codes over the full remaining matrix once and reuses `cohen_kappa` — an equivalent shortcut because the unweighted kappa is level-set invariant (PROVEN per-pair in the oracle; the corresponding mutant is an unkillable identity, documented). Deviations: a pair with pe = 1 and `chanceP ≤ 0` — reachable on valid data with disjoint rater level sets, where R silently emits NaN — are explicit errors. Exact-Fraction oracle anchors L1–L5 (value 59/99, chanceP 17189/125000, z 4.719794049843912; six-pair fixture 430543/982080; listwise-drop pins 1/3 vs mutant 91/300), a 5-mutant EXECUTED kill map (mean→sum, disrater diagonal, npairs factor, pair-loop bound, listwise drop), and an MC-500 rater-permutation invariance test.
- `kripp_alpha` Krippendorff's alpha for inter-rater agreement (CRAN irr 0.85 `kripp.alpha()` `R/kripp.alpha.R` — READ and normative; Krippendorff 1980 NOT READ, cited as method origin only): coincidence matrix over unordered rater pairs per subject with the irr divisor quirk preserved verbatim (`mc = #nonmissing − 1` per column ONLY when the matrix contains any missing value, else 1 — complete-data alpha differs from the m−1 convention), diagonal increment `2/mc`, mirror by assignment, `nmatchval` as total cell mass, and all four distance metrics (nominal, ordinal with half-endpoint coincidence-row-sum weights, interval, ratio) feeding `α = 1 − (nmatchval−1)·Σ(utcm·δ²)/Σ(nc_c·nc_k·δ²)`. Fewer than 2 observed levels yields α = 1 (R line 45). Documented deviations: all-missing matrix, infinities, and ratio level pairs summing to zero are explicit errors (R would return α = 1, propagate, or emit Inf/NaN). Exact-Fraction oracle anchors K1–K4 (irr man-page matrix: nominal 113/152, ordinal 108577/133160, interval 951/1120, ratio 18222619/22852465, nmv = 40; no-NA quirk pin 43/72 vs the m−1 mutant's 11/18), a 6-mutant EXECUTED kill map (diagonal 1/mc, mc always m−1, ordinal full weights, interval |δ|, nmv off-diagonal only, num×nc products), and an MC-500 rater/subject permutation-invariance test.
- `icc` intraclass correlation coefficients for inter-rater reliability, the full Shrout-Fleiss taxonomy (CRAN irr 0.85 `icc()` `R/icc.R` — READ and normative; Shrout & Fleiss 1979, McGraw & Wong 1996, Bartko 1966 NOT READ, cited as model origins only): `model` oneway/twoway × `type` consistency/agreement × `unit` single/average from one-pass ANOVA mean squares (MSr, MSw, MSc, MSe with sample-variance divisor n−1), the F test of H0: icc = r0 (two-way agreement uses the Satterthwaite df with the R quirk that both units' confidence bounds plug the estimate into the nr-scaled a,b form — icc.R lines 139-141 preserved verbatim), and unclamped confidence bounds. Rows with NaN are dropped listwise (R `na.omit`); infinities rejected; degenerate zero-variance and icc=1 pivots return explicit errors instead of leaking non-finite output. Exact-Fraction oracle anchors on Shrout-Fleiss Table 2 (all six coefficients: 448/2703, 920/1287, 184/635, 1792/4047, 3680/4047, 736/1187 plus scipy F-distribution CI pins), a 6-mutant EXECUTED kill map (MSw divisor, quantile df order, agreement denominator, r0 in F, dimension map, CI plug-in), and an MC-500 test of subject/rater permutation invariance plus the Spearman-Brown single↔average bridge for all three families.
- `fleiss_kappa` Fleiss' multi-rater kappa for nominal agreement with the exact (Conger) chance-agreement variant (CRAN irr 0.85 `kappam.fleiss()` `R/kappam.fleiss.R` — READ and normative; Fleiss 1971 and Conger 1980 NOT READ, cited as model origins only): classification-table agreement `agreeP = (1/m)Σᵢ(Σⱼtᵢⱼ² − nr)/(nr(nr−1))`, classic chance `Σⱼpⱼ²` vs exact `Σⱼpⱼ² − (1/nr)Σⱼs²ⱼ` (sample variance over per-rater category proportions; algebra verified against R's `apply(rtab,2,var)` form), Fleiss' large-sample z test and category-wise kappas (classic mode; NaN for empty categories, matching R's 0/0), and listwise row drop for missing ratings. API deviations documented: index codes 0..k-1 with explicit/inferred k, negative-or-NaN = missing, error on degenerate `1 − chanceP = 0` (R returns NaN). Exact-Fraction oracle anchors FK1–FK5 (classic κ = 139/399, exact κ = 37/102, category κ = [1/21, 31/91, 43/63]), a 6-mutant EXECUTED kill map (row-vs-column chance sums, missing-as-category, variance-sign, pjk-centering), and an MC-500 subject/rater permutation-invariance test.
- `bratt_mm` Bradley-Terry model with ties fitted by MM (VGAM 1.1-14 `bratt()` family, `R/family.categorical.R` — READ and normative; Bradley & Terry 1952 NOT READ, cited as model origin): `P(i>j) = αᵢ/(αᵢ+αⱼ+α₀)`, `P(tie) = α₀/(αᵢ+αⱼ+α₀)` with a hand-derived supporting-hyperplane MM ascent (same pattern as the crate's `bradley_terry_mm`) and a joint reference rescale of α AND α₀ (likelihood-preserving; verified identity `Σ wins + T = Σ n_ij`). This is the additive-α₀ ties model, NOT Rao-Kupper/Davidson (neither read; disambiguation only). Contract: fractional weighted counts accepted, symmetric ties matrix required, tie-free data rejected (use `bradley_terry_mm`; an API contract, not VGAM behavior), zero-win contestants rejected, n capped at 10000 (O(n²) guard). Exact-Fraction oracle anchors B1–B4 (iteration-1 pins `[1, 27/40, 3/4]`, α₀ = 9/14), a 5-mutant EXECUTED kill map (incl. a tol-separated convergence anchor killing an α₀-blind convergence check), and an MC-500 log-likelihood dominance test.
- `predict_rating` / `predict_rating_multi` game-outcome prediction from fitted ratings (CRAN PlayerRatings 1.1-0 `predict.rating` `R/ratings.R` lines 1056–1133 — READ and normative; no journal paper exists for this dispatch, CRAN package provenance only): Elo logistic branch, deviation-shrunk Glicko/Glicko-2/Stephenson branch (`qip3 = 3(ln10/400π)²`, joint shrink over BOTH players' squared deviations), and multi-player EloM branch (`(rating − rowmean)/40`, na.rm row means, optional min-tie placing ranks with NaN kept). R semantics preserved: strict `games < tng` unrated cutoff, `trat` replacement of ALL missing extracted values (unmatched, low-games, stored-NA), `pred >= thresh` binarization with NaN propagation. REDUCED-SCOPE vs R: index-based (−1 = unmatched; caller does name matching), per-game/scalar gamma only. Exact-oracle fixtures P1–P9 and a 7-mutant EXECUTED kill map (incl. both branches' tng comparisons).
- `fide_rating` FIDE-style Elo ratings (CRAN PlayerRatings 1.1-0 `fide()` `R/ratings.R` lines 125–272 + `kfide()` lines 959–972 — READ and normative; no journal paper exists for this variant, CRAN package provenance only): per-period batch Elo with the kfide K-factor schedule (K = kv[0] elite / kv[1] ≥30 games / kv[2] novice, evaluated from PERIOD-START state), sticky elite flag set from POST-update ratings ≥ 2400, and per-player running mean of POST-update opponent ratings. REDUCED-SCOPE vs R: no status/history frames, kfide-only K schedule, self-play rejected, thresholds 30/2400 hard-coded. kv=(k,k,k) reduces bitwise to `elo_rating(kfac=k)` (MC-500 anchor); exact-oracle fixtures F1–F5 and a 5-mutant EXECUTED kill map.
- `metrics_rating` prediction-quality metrics for binary-outcome forecasts (binomial deviance on capped predictions, RMSE/MAE on raw predictions, optional 0.5-baseline scaling), a Rust reimplementation of CRAN PlayerRatings 1.1-0 `metrics()` with its cap quirk and elementwise NaN semantics preserved; exact-oracle anchor tests, mutation-kill map, and MC-500 invariants.
- **Multiplayer Elo rating (CRAN PlayerRatings 1.1-0 `elom()`
  `R/ratings.R` lines 739–932 + `elom_c` C kernel `src/ratings.c`
  lines 45–80, and `kriichi` K-factor lines 1006–1020 — all READ and
  normative; no journal paper exists for this system, CRAN package
  provenance only)**: `elom_rating` scores nn-seat events (empty seats
  as player −1/NaN score) with rank base scores, a per-period single
  update `K·(actual − expected)` where expected sums
  `(r_p − event mean rating)/40`, and either a constant K or the
  kriichi experience-decay `max(kv, 1 − (1−kv)·games/gv)`. Faithfully
  reproduces the R quirk that partial events shrink the ORIGINAL base
  exactly once regardless of empty-seat count (R:855-866 `sbase <-
  basev` resets inside the loop). REDUCED-SCOPE vs R: player −1 ⟺ NaN
  score jointly enforced, sorted periods required, in-event duplicate
  players rejected, kriichi bounds `gv > 0`, `0 < kv ≤ 1`. Rust core
  `mlsirm_core::scaling::elom_rating` with exact-value anchors E1–E9
  (dyadic rationals; hand-derived oracle executed against the R
  semantics) and a 500-rep Monte-Carlo invariance test; 6 mutation
  kills executed (K-scaling, event-mean, cumulative-shrink, kriichi
  games-timing, tie-rank, per-event-update).

- **Stephenson rating system (CRAN PlayerRatings 1.1-0 `steph()`
  `R/ratings.R` lines 591–737 + `stephenson_c` C kernel
  `src/ratings.c` lines 157–202 — both READ and normative; no journal
  paper exists for this system, which the package attributes to Alec
  Stephenson's winning entry in the 2010 Kaggle chess-rating contest,
  NOT independently verifiable beyond the package provenance).** New
  Rust core `mlsirm_core::scaling::stephenson_rating` extending Glicko
  with a per-game neighborhood variance term (`ngames·hval²`), a
  per-game bonus `bval/100` added to each played game's score on BOTH
  sides, a participants-only lambda drift toward opponents' ratings
  (`(λ/100)·Σ(r_opp−r_self)/ngames`), and `(lag+1)·cval²` per-period
  deviation-variance inflation clamped at `rdmax²` (all formulas
  line-cited to the READ R/C source in the code header). Supports
  prior-run continuation via `init_games`/`init_lag` and per-game
  white-advantage `gamma`. PyO3 binding + thin NumPy wrapper
  `fast_mlsirm.stephenson_rating` (defaults `init=(2200, 300)`,
  `cval=10`, `hval=10`, `bval=0`, `lambda_=2`, `rdmax=350` matching
  PlayerRatings). Anchored against an EXECUTED faithful oracle port of
  the R driver + C kernel (heterogeneous-init, two-period draw/lag,
  full-knobs, rdmax-clamp, bval-symmetry, and λ=0 contrast fixtures
  pinned to 1e-12; five mutation kills EXECUTED: bval drop, λ sign
  flip, per-game hval scaling drop, `(lag+1)→lag`, opponent-g→own-g).
- **Glicko-2 rating system (Glickman's 2022 *Example of the Glicko-2
  system* note READ — worked example reproduced; CRAN PlayerRatings
  1.1-0's `glicko2()` `R/ratings.R` + `glicko2_c` C kernel source READ;
  Glickman's 2001 J. Appl. Statist. derivation paper NOT READ, cited as
  the origin per both READ sources).** New Rust core
  `mlsirm_core::scaling::glicko2_rating` with batch-per-period updates on
  the Glicko-2 scale, per-player rating volatility via Glickman's Step-5
  Illinois iteration (epsilon 1e-6, endpoint A; DERIVED and documented:
  Glickman's `f(x)` equals `-1/2` the derivative of PlayerRatings'
  penalized negative log-likelihood, so the Illinois root matches R's
  optimum), participant-only pre-period variance inflation
  `phi^2 <- min(phi^2 + lag * sigma^2, (q rdmax)^2)` (Glicko-2 uses `lag`,
  not Glicko-1's `lag+1`; R source comment pinned), `tau == 0` volatility
  freeze, volatility ceiling `q * rdmax`, per-game white advantage
  `gamma`, and PlayerRatings W/D/L and lag bookkeeping. Documented
  R-vs-note deviation: the note applies Step 6 to idle players every
  period; PlayerRatings (and this port) defer idle growth via
  `lag * sigma^2` at next participation. Python wrapper
  `fast_mlsirm.glicko2_rating(games, n_players, init=(2200, 300, 0.15),
  gamma=None, tau=1.2, rdmax=350)` returns a `Glicko2Result` dataclass and
  inherits the Elo/Glicko period-label fidelity contract. Anchored to an
  executed float64 oracle (Glickman worked-example anchor with
  heterogeneous init, two-period inflation/lag/idle pins, rdmax and
  volatility-ceiling clamps, gamma, unsorted-period, fractional-score +
  tau-0, and return-after-idle fixtures); nine executed mutation kills
  (own-g swap, lag off-by-one, variance-clamp drop, volatility-clamp
  drop, skipped volatility update, stale-sigma inflation, Illinois
  endpoint swap, gamma sign, rating-before-deviation order).
- **Glicko rating system (Glickman's *The Glicko system* technical note
  READ — worked example reproduced exactly; CRAN PlayerRatings 1.1-0's
  `glicko()` `R/ratings.R` + `glicko_c` C kernel source READ; Glickman's
  1999 Applied Statistics derivation paper NOT READ, cited as the origin
  per both READ sources).** New Rust core
  `mlsirm_core::scaling::glicko_rating` with batch-per-period updates,
  per-player rating deviations, participant-only pre-period deviation
  inflation `RD = min(sqrt(RD^2 + (lag+1) c^2), rdmax)` (Step 1b), the
  Step 2 update with opponent-g weighting and new-variance rating step,
  per-game white advantage `gamma`, and PlayerRatings W/D/L and lag
  bookkeeping. Documented non-identity: Glicko does NOT conserve the
  rating sum (asymmetric opponent-g weighting; pinned by test). Documented
  divergences: self-play rejected, no `status` carry-in — per-player
  `init_rating`/`init_dev` arrays with results for ALL `0..n` players.
  Python wrapper `fast_mlsirm.glicko_rating(games, n_players,
  init=(2200, 300), gamma=None, cval=15, rdmax=350)` returns a
  `GlickoResult` dataclass and inherits the Elo period-label fidelity
  contract (integer-dtype lossless path; dtype-derived float bound).
  Anchored to an executed float64 oracle (Glickman worked-example anchor
  with heterogeneous RDs, two-period inflation/lag/idle-player pins,
  rdmax-clamp, gamma, unsorted-period, and fractional-score fixtures);
  seven executed mutation kills (opponent-g swap, inflation off-by-one,
  clamp drop, stale-variance update, missing q^2, gamma sign,
  all-player inflation).

- **Elo rating system (Elo, 1978, as implemented by CRAN PlayerRatings
  1.1-0's `elo()` — `R/ratings.R` + `elo_c` C kernel source READ; Elo's
  1978 book NOT READ, cited as the origin per PlayerRatings).** New Rust
  core `mlsirm_core::scaling::elo_rating` with batch-per-period updates
  (all expected scores within a rating period use the period-start
  ratings), per-game white advantage `gamma`, and PlayerRatings W/D/L and
  lag bookkeeping (W/D/L only for scores exactly 1/0.5/0; lag counts
  periods since last appearance). Periods may be unsorted (grouped by
  ascending label, matching R `split()`); `E_w + E_b = 1` holds
  identically for any gamma, so rating sums are conserved at `n * init`.
  Documented divergences: self-play rejected, scalar K factor only, no
  `status` carry-in. Python wrapper `fast_mlsirm.elo_rating(games,
  n_players, init=2200, kfac=27, gamma=None)` returns an `EloResult`
  dataclass. Anchored to an executed exact-rational oracle (single- and
  two-period exact-fraction fixtures, float regression, closed-form
  nonzero-gamma pin); five executed mutation kills (sequential-update,
  black-score flip, gamma sign, lag reset, logistic divisor).

- **Circular-triads consistency test and Kendall's coefficient of
  agreement u (Kendall & Babington Smith, 1940, as implemented by eba
  1.10-0's `circular()` / `kendall.u()` — eba source READ; the 1940
  Biometrika paper and Alway's (1962) exact tables NOT READ, cited as
  origins per eba's manual pages).** New Rust cores
  `scaling::circular_triads` (number of circular triads
  `T = C(n,3) - sum_j C(d_j, 2)`, maximum, consistency coefficient
  `zeta = 1 - T/T_max`, and a null test that is EXACT for `n <= 10`
  via embedded null distributions — dyadic-rational p-values — and a
  continuity-corrected chi-square for `n >= 11`) and
  `scaling::kendall_u` (agreement between `m >= 3` judges: `Sigma`,
  `u = 2*Sigma/(C(m,2)*C(n,2)) - 1`, minimum attainable `u`, and an
  upper-tail chi-square test whose RAW statistic may be negative under
  the continuity correction; only the p-value clamps). Documented
  divergences from eba: `n = 2` tournaments and malformed/incomplete
  input are rejected, and every pair must have the same number of
  judges (eba reads `m` from the first pair only). Python wrappers
  `fast_mlsirm.circular_triads` / `kendall_u` return
  `CircularTriadsResult` / `KendallUResult` dataclasses.

- **Luce-choice top-1 estimation via Luce Spectral Ranking (Maystre &
  Grossglauser, 2015, as implemented by choix 0.4.1's `lsr_top1` /
  `ilsr_top1` — choix source READ; the paper itself NOT READ, cited as
  the algorithm origin per choix's docstrings).** New Rust cores
  `scaling::lsr_top1` (one-shot) and `scaling::ilsr_top1` (iterative
  MLE) estimate Luce/Plackett-Luce log-worths from top-1 choice data
  (`(winner, losers)` observations, CSR layout): each observation
  accrues rate `1/(sum of choice-set worths)` on every loser-to-winner
  edge (plus `alpha` regularization), and the centered log stationary
  distribution of that chain is the estimate. Python wrappers
  `fast_mlsirm.lsr_top1` / `ilsr_top1` take iterables of
  `(winner, losers)` pairs and return `LsrResult`. Three documented
  divergences from choix: empty loser sets are rejected (choix silently
  no-ops them), a winner in its own loser set is rejected (choix
  silently inflates the denominator), and duplicate losers are rejected
  (choix double-counts the edge). Single-loser observations bit-match
  `lsr_pairwise` on the induced win matrix (regression-pinned).
- **Plackett-Luce ranking estimation via Luce Spectral Ranking (Maystre &
  Grossglauser, 2015, as implemented by choix 0.4.1's `lsr_rankings` /
  `ilsr_rankings` — choix source READ; the paper itself NOT READ, cited as
  the algorithm origin per choix's docstrings).** New Rust cores
  `scaling::lsr_rankings` (one-shot) and `scaling::ilsr_rankings`
  (iterative MLE) estimate Plackett-Luce log-worths from full or partial
  rankings (best first, CSR layout): each ranking is a sequence of Luce
  choices, accruing rate `1/(sum of remaining ranked worths)` on every
  loser-to-winner edge (plus `alpha` regularization), and the centered log
  stationary distribution of that chain is the estimate. Python wrappers
  `fast_mlsirm.lsr_rankings` / `ilsr_rankings` take lists of rankings and
  return `LsrResult`. Three documented divergences from choix: rankings
  shorter than 2 items are rejected (choix silently no-ops them),
  within-ranking duplicates are rejected (choix accepts them when the
  chain stays connected), and negative indices are rejected before the
  unsigned cast (Python's would silently wrap). Exact rational anchors
  from an executed exact-Fraction/mpmath oracle (full- and
  partial-rankings fixtures — the partial fixture is the only one that
  can see a wrong all-items denominator), length-2 equivalence with
  `lsr_pairwise` pinned bit-exact, I-LSR fixed-point and iteration-count
  pins, five executed mutation kills, and a 500-replication Monte-Carlo
  recovery harness (`--ignored`) with a measured bound.
- **Rank Centrality paired-comparison estimator (Negahban, Oh, & Shah, 2017,
  as ported by choix 0.4.1's `rank_centrality` — choix source READ; the
  paper's discrete-time max-degree walk is NOT what choix computes and only
  the choix continuous-time win-ratio chain is implemented).** New Rust core
  `scaling::rank_centrality` builds the Markov chain whose loser-to-winner
  rates are the regularized win ratios
  `(alpha + wins[i,j]) / (2*alpha + wins[i,j] + wins[j,i])` and returns the
  centered log stationary distribution via the shared Gaussian-elimination
  stationary solver (`statdist_params`, extracted from the LSR pass), with
  explicit `Err` on disconnected graphs at `alpha = 0` and on overflowing
  counts or ratio denominators (choix silently degrades there). Exposed as
  `fast_mlsirm.rank_centrality` returning `LsrResult`. Exact-Fraction oracle
  anchors (3x3, 4x4, alpha 0 and 1/2, one-sided, disconnected-with-alpha)
  cross-checked against pip choix 0.4.1 to 2e-16; exact scale invariance
  pinned at `alpha = 0`; six mutation kills executed (transposed ratio,
  missing ratio transform, half-updated denominator, dropped normalization,
  dropped centering, denominator-c-only).
- **Luce Spectral Ranking (LSR) and iterative LSR (I-LSR) paired-comparison
  estimators (Maystre & Grossglauser, 2015, as implemented by choix 0.4.1's
  `lsr.py` dense pairwise path — source READ; Maystre & Grossglauser, 2015
  NOT READ, cited as described by choix)** (`fast_mlsirm.lsr_pairwise`,
  `fast_mlsirm.ilsr_pairwise`; in Rust `mlsirm_core::scaling::lsr_pairwise`
  / `ilsr_pairwise`): builds the LSR Markov chain (rate `c/(w_i + w_j)` on
  each loser→winner edge plus `alpha` everywhere as a regularizer) and
  returns the centered log of its stationary distribution (scaled to sum n,
  choix `statdist` convention). The stationary solve uses Gaussian
  elimination with partial pivoting plus positivity, sum, and residual
  guards; disconnected comparison graphs at `alpha = 0` and overflow from
  huge counts or alpha raise errors instead of returning NaN (divergence
  from choix, which can emit NaN there). I-LSR at `alpha = 0` converges to
  the Bradley–Terry MLE and agrees with `bradley_terry_mm` (verified to
  4e-19 on the oracle fixtures); for `alpha > 0` the two regularization
  paths deliberately differ (chain-rate vs Dirichlet-MAP, each per its
  source). Pinned against an EXECUTED exact-Fraction/mpmath oracle
  cross-checked with pip choix 0.4.1 (≤ 2.2e-13); six mutation kills
  executed (chain transpose, dropped diagonal subtraction, dropped
  centering, sum-n→sum-1 normalization via weights pins, denominator
  collapse via I-LSR pins — one-shot-unobservable limitation documented —
  and tol·n→tol via a fixture whose iteration counts separate).
- **Bradley–Terry maximum-likelihood paired-comparison worths via the MM
  algorithm (Hunter, 2004, as implemented by choix 0.4.1's `opt.mm` pairwise
  path; Maystre — source READ; Bradley & Terry, 1952 and Hunter, 2004 NOT
  READ, cited as described by choix)** (`fast_mlsirm.bradley_terry_mm`; in
  Rust `mlsirm_core::scaling::bradley_terry_mm`): fits centered log-worths
  from an n×n win-count matrix (`wins[i, j]` = times i beat j; non-integer
  counts accepted as a DERIVED weighted extension) with choix's
  regularization `alpha`, exp-scale weights normalized to sum n, and the
  L1 convergence rule `sum |new − prev| ≤ tol·n` over consecutive updates.
  All-zero matrices are rejected for every alpha (deliberate divergence from
  choix's uniform fallback at alpha > 0); zero-wins items at alpha = 0 and
  Ford-condition violations (an unbeaten item) raise instead of returning a
  bogus fit. Pinned against a 50-digit mpmath oracle cross-checked with
  choix itself (max diff ≤ 1.4e-12), including an exact ±ln(3)/2 closed-form
  2×2 anchor, an alpha = 0.5 MAP anchor, and an iterations == 18 convergence
  pin; five mutation kills executed (winner accumulation, denominator
  symmetry, centering, weight normalization, tol·n semantics) plus a 500-rep
  Monte-Carlo recovery test (`#[ignore]`).
- **Thurstone Case V paired-comparison scaling (Thurstone, 1927, as
  implemented by psych's `thurstone()`; Revelle, 2025 — source READ)**
  (`fast_mlsirm.thurstone_case_v`; in Rust
  `mlsirm_core::scaling::thurstone_case_v`): scales n objects from an n×n
  choice-probability matrix (`choice[i, j]` = P(column j preferred over row
  i)) via `scale_j = colmean(qnorm(choice))_j − min`, fitted model
  `Phi(scale_j − scale_i)`, residuals, and psych's goodness of fit
  `1 − sse/ssc` over the full model matrix (pinning the psych *code*
  behavior; the .Rd "lower off diagonal" prose is stale). Entries must be
  strictly in (0, 1) — a deliberate safety divergence from psych, whose
  direct path admits infinite normal quantiles. Pinned against a 50-digit
  mpmath oracle on asymmetric, exactly-consistent, and intransitive
  fixtures.
- **Composite linking (Holland & Strawderman, 2011; as cited by Albano,
  2016, JSS 74(8), eqs. 31-32)** (`fast_mlsirm.composite_linking`; in Rust
  `mlsirm_core::equating::composite_linking`): weighted average of H
  component conversion tables over a shared x grid. With per-component
  linear slopes supplied, applies the symmetric Holland-Strawderman weight
  adjustment `W_h = w_h (1 + a_h^p)^(-1/p) / sum(...)` (eq. 32), which for
  linear components makes the composite of forward links the exact
  functional inverse of the composite of inverse links (pinned by an exact
  round-trip test). Without slopes, raw weights are normalized
  (`W_h = w_h / sum(w)`) — a documented deviation from the R `equate`
  package's un-normalized non-symmetric path (identical iff weights sum
  to 1). Exact-fraction oracle pins, 5 executed mutation kills (dropped
  adjustment, exponent sign flip, skipped normalization, `a^p → a*p` at
  p=2, weight/table zip reversal), and a 500-rep Monte-Carlo round-trip
  invariant (`#[ignore]`).
- **Nominal weights mean equating (Babcock, Albano, & Raymond, 2012; as
  restated by Albano, 2016, JSS 74(8), eq. 42)**
  (`fast_mlsirm.nominal_weights_mean_equate`; in Rust
  `mlsirm_core::equating::nominal_weights_mean_equate`): NEAT-design mean
  equating for very small samples — the Tucker regression slopes are
  replaced by the nominal-weights effective-length ratios
  `gamma1 = k_x/k_v`, `gamma2 = k_y/k_v` (item counts), synthetic means
  follow Albano (2016) eqs. 37-38, and the conversion is the slope-1 mean
  shift `yx(x) = x + (mu_sY - mu_sX)` (eq. 10). Synthetic variances
  (eqs. 39-40, N-denominator moment convention, not the R package's N-1
  sample variances) are reported but do not enter the conversion. Oracle:
  exact-Fraction hand computation plus an executed cross-check against the
  method authors' R package `equate` 2.0.8 (KBneat intercept
  0.5833490108414594). The 2012 EPM article is paywalled and was NOT read;
  the method is implemented from Albano (2016) and the authors' own R
  source, both read. Five mutation kills executed (gamma swap, w1/w2 swap,
  anchor-mean-difference sign, non-unit slope, dropped w1*w2*g^2*d^2
  variance term).
- **Circle-arc small-sample equating (Livingston & Kim, 2008, ETS
  RR-08-39)** (`fast_mlsirm.circle_arc_equate`,
  `fast_mlsirm.circle_arc_middle_anchor`; in Rust
  `mlsirm_core::equating::{circle_arc_equate, circle_arc_middle_anchor}`):
  constrains the equating curve through two prespecified end-points and an
  empirically estimated middle point. Method 1 fits a circle arc directly
  through the three points (circumcenter eqs. 3-4, radius eq. 5, arc
  branch chosen by the middle point's position relative to the center);
  Method 2 (the source's most accurate small-sample method) decomposes the
  curve into the linear component `L(x)` through the end-points plus an
  arc fitted to the transformed points `y* = y - L(x)`. Collinear points
  degenerate to the line. `circle_arc_middle_anchor` computes the
  anchor-design middle point `y2 = m_YB + (s_YB/s_VB)(m_VA - m_VB)`
  (eq. 9). Reduced scope: scores must lie within the end-points (the
  source's below-lower-endpoint linear extension is not implemented).
  Pinned by the paper's worked example (center `(40, -15)`, `r^2 = 1625`;
  Method-2 transformed center `(12.5, -13)`, `r^2 = 901/4`), a Table-1
  anchor pin, an exact minus-branch fixture, and a 500-rep randomized
  anchor-recovery check, all validated against an exact-Fraction oracle.

- **Pass-fail reliability from parallel half-tests (Woodruff & Sawyer,
  1988, AERA paper, ERIC ED292877)** (`fast_mlsirm.woodruff_sawyer_sb`,
  `fast_mlsirm.woodruff_sawyer_normal`; in Rust
  `mlsirm_core::classification::{woodruff_sawyer_sb,
  woodruff_sawyer_normal}`): estimates the full-test agreement `theta*`
  and coefficient `phi*` (Cohen's kappa for the symmetric 2x2 pass-fail
  table) from a single administration split into parallel halves. The SB
  method symmetrizes the half-test 2x2 table's off-diagonal, computes
  `phi = 1 - pi01/(pq)` (eq. 1), steps up by Spearman-Brown
  `phi* = 2 phi/(1+phi)` (eq. 5), and reconstructs the full-length table
  (eq. 8); the normal method steps up the half-test correlation
  (`r_SB = 2r/(1+r)`), models parallel full forms as bivariate normal, and
  evaluates the joint fail-fail cell with the crate's BVN quadrature.
  Pinned by exact rational fixtures, a Sheppard-orthant exact anchor, and
  the paper's Table 4 values; six mutation kills executed. Per the source
  (pp. 9-10), `phi*` from the SB method is positively biased when the
  halves are not strictly parallel.
- **Livingston's criterion-referenced reliability k^2 and correlation
  k(X, Y) (Livingston, 1972, AERA paper, ERIC ED069624)**
  (`fast_mlsirm.livingston_k2`, `fast_mlsirm.livingston_correlation`; in
  Rust `mlsirm_core::classification::{livingston_k2,
  livingston_correlation}`): the classical-test-theory analogues of
  reliability and correlation with moments taken about a criterion
  (cut) score instead of the mean, `D^2(X) = var + (mean - cut)^2`,
  `k^2 = (rho^2 var + (mean-cut)^2) / D^2(X)`, and
  `k(X,Y) = D(X,Y)/sqrt(D^2(X) D^2(Y))`, with Spearman-Brown test-length
  projections applied to `k^2` itself. The conversion form is an
  algebraic reconstruction from the source's Table 1 expectation
  definitions; `k^2` is NaN only in the exact degenerate case (scores all
  exactly equal to the cut, detected element-wise, or `var == 0 &&
  mean == cut`), and returns the formula limit 1 when the squared
  criterion offset overflows f64 with finite variance (the correlation
  rejects that overflow with an error); fractional Spearman-Brown lengths are a
  disclosed continuous extrapolation. Exact-fraction anchors (k2 = 5/6,
  SB(2) = 10/11, sign-flip k = 5/7 with norm rho = -1, asymmetric-offset
  k = 22/(7 sqrt(10))), equality-iff-mean=cut and zero-variance property
  pins, error contracts, and a 500-rep Monte Carlo recovery check
  (`#[ignore]`).
- **Brennan-Kane index of dependability Phi(lambda) for mastery tests
  (Kane & Brennan, 1977, ACT Technical Bulletin No. 28, ERIC ED185076,
  eq. 33)** (`fast_mlsirm.phi_lambda`; in Rust
  `mlsirm_core::gtheory::phi_lambda`): the criterion-referenced
  dependability coefficient theta(d) = Phi(lambda) for a one-facet random
  `p x i` design at a cutting score `lambda`, built on the module's
  `gtheory_pi` ANOVA. The `(Xbar - lambda)^2` signal is estimated with a
  derived unbiased plug-in that subtracts `varhat(Xbar)` computed from the
  RAW (unclamped) variance components, while `sigma^2(Delta')` and the
  `sigma^2(p)` numerator keep the module's clamped-component policy; the
  signal is left unclamped, so estimates may fall below the lambda-free
  `dependability` (finite-sample behavior, documented). TB-28 defers
  estimation to Brennan & Kane (1977a, JEM), which was not read; the
  estimator is derived and adversarially verified independently.
- **Subkoviak single-administration coefficient of agreement (Subkoviak,
  1976, ERIC ED120229 / JEM 13(4))** (`fast_mlsirm.subkoviak_agreement`; in
  Rust `mlsirm_core::classification::subkoviak_agreement`): per-person and
  group coefficients of agreement, marginal chance agreement, and Cohen's
  kappa for mastery classifications under the simple binomial true-score
  model, with the regression estimate of the item-domain proportion
  (Eq. 16) and optional KR-21 reliability derived from the data with the
  population (ddof = 0) variance. Supports multi-category criteria
  (Eqs. 19-22); mastery convention is score `>= C`, verified against
  Table 1 of the read source (its Eq. 4 OCR prints `>`). The compound
  binomial refinement (Eqs. 12-14) and Lord's (1959) distribution-free
  estimate (Eq. 17) are excluded because they defer to sources not read.
  Exact-fraction oracle pins from the paper's Table 1 fixture; five
  executed mutation kills (category boundary, P(i) squaring, chance-term
  aggregation, KR-21 ddof, regression-weight swap).
- **Hanson-Brennan compound-binomial classification consistency and accuracy
  (Hanson, 1991, ACT Research Report 91-5)** (`fast_mlsirm.hanson_brennan`,
  `fast_mlsirm.hanson_brennan_from_params`; in Rust
  `mlsirm_core::classification::hanson_brennan` /
  `hanson_brennan_from_params`): single-administration decision consistency,
  accuracy, sensitivity, specificity, and Cohen's kappa for
  number-correct cut scores under a four-parameter beta true-score
  distribution with Lord's two-term approximation to the compound binomial
  conditional error model. The data path estimates Lord's k from the score
  mean/variance and reliability (Hanson, 1991, Eq. 6), recovers the first
  four true-score moments by the HB.tsm recursion (Eqs. 7-8), fits the
  four-parameter beta by the method of moments with a two-parameter
  failsafe (identical branch structure to `livingston_lewis`); the params
  path accepts explicit (l, u, alpha, beta, k). The conditional fail CDF
  uses a derived closed form
  `BinCdf(cut-1;K,p) - k p(1-p) [b(cut-1;K-2,p) - b(cut-2;K-2,p)]`,
  verified as an exact polynomial identity against Lord's term-by-term
  definition in the oracle. Pinned against an exact-Fraction stdlib oracle
  (params fixtures at 1e-12; a genuine negative-k 4P data fixture with both
  beta shapes < 1 at 1e-7); five mutation kills executed.
- **Two-stage adaptive testing (Betz & Weiss, 1973, Research Report 73-4;
  Betz & Weiss, 1974, Research Report 74-4)** (`fast_mlsirm.two_stage_route`,
  `fast_mlsirm.two_stage_score`; in Rust
  `mlsirm_core::exposure::two_stage_route` / `two_stage_score`, PyO3
  `py_two_stage_route` / `py_two_stage_score`): routing-test scoring via the
  truncated normal-ogive ability estimate theta-hat =
  Phi^-1(((x'/m) - c) / (1 - c)) / a-bar + b-bar (Equation 2; perfect scores
  truncate to m - 1/2, chance-or-below scores to c*m + 1/2), assignment of
  the measurement test whose mean difficulty is closest to the routing
  estimate (minimum absolute difference; ties break to the lowest index, a
  derived convention), and the item-count-weighted composite
  (m1*theta1 + m2*theta2)/(m1 + m2) (Equation 3). The scoring entry point
  re-derives the routing assignment and refuses a mismatched
  `administered` index so second-stage scores are never combined with the
  wrong measurement test's parameters. Anchored on the reconstructed
  Appendix B routing table of Research Report 74-4 and an exact-Fraction
  oracle through the p-computation; both subtests require m*(1-c) > 1 for
  distinct truncation endpoints.
- **Pyramidal adaptive testing (Larkin & Weiss, 1974, Research Report
  74-3)** (`fast_mlsirm.pyramidal_administer`; in Rust
  `mlsirm_core::exposure::pyramidal_administer`, PyO3
  `py_pyramidal_administer`): deterministic single-examinee replay of the
  classic up-one/down-one equal-offset pyramidal ("branched") design — items
  in a triangular structure ordered by difficulty (stage s holds s items,
  n(n+1)/2 total), a correct response routing to the harder stage-(s+1)
  neighbour and an incorrect response to the easier — with Larkin & Weiss's
  six scoring methods: number-correct, mean difficulty attempted, mean
  difficulty correct (NaN when indeterminate), final-item difficulty, the
  hypothetical (n+1)th-item "final difficulty score" (computed only when the
  caller supplies the next-stage difficulties; the paper's pool-specific
  column-mean construction is out of scope), and Hansen's all-item score as
  described by Larkin & Weiss (verified against the printed 15-stage 0–240
  range). Routing recurrence and all-item stage scores are DERIVED from the
  source prose (labelled in the module comment); exact-fraction oracle
  anchors, checked-arithmetic overflow guards, and a 500-rep Monte-Carlo
  structural invariant test (`#[ignore]`).
- **Weiss stradaptive (stratified-adaptive) test administration (Weiss, 1973,
  Research Report 73-3)** (`fast_mlsirm.stradaptive_administer`; in Rust
  `mlsirm_core::exposure::stradaptive_administer`, PyO3
  `py_stradaptive_administer`): deterministic single-examinee replay of
  Weiss's stratified-adaptive design — an item pool partitioned into S ≥ 2
  difficulty strata, up-one-stratum after a correct response and
  down-one-stratum after an incorrect response (edge-clamped, with a DERIVED
  fallback to the last administered stratum when the clamped target is
  exhausted), terminating on a ceiling stratum (≥ min_items administered and
  proportion correct ≤ chance), pool exhaustion, or max_items. Reports
  ceiling / basal / highest-non-chance strata, Weiss's ten ability scores
  m1–m10 (NaN when indeterminate; score 7 interpolates between adjacent
  stratum mean difficulties with side-dependent steps), and a consistency
  index (population variance of the score-9 stratum set; DERIVED — defined
  verbally in the report without a printed numeric anchor). The primary
  source was READ (ERIC ED084301); routing and scores are pinned by the
  report's William W. protocol (Fig. 2 / Appendix A) and its five printed
  score-7 cases plus synthetic below-chance anchors that discriminate the
  lower-step branch (the printed cases alone do not). Score 5's
  extrapolated-next-stratum variant for exhausted pools and free-response
  (chance = 0) termination are deliberately out of scope.
- **Lord self-scoring flexilevel testing (Lord, 1970, RB-70-43; Lord, 1971,
  RB-71-6)** (`fast_mlsirm.flexilevel_administer` /
  `fast_mlsirm.flexilevel_score_distribution`; in Rust
  `mlsirm_core::exposure::flexilevel_administer` /
  `flexilevel_score_distribution`, PyO3 `py_flexilevel_administer` /
  `py_flexilevel_score_distribution`): deterministic replay of Lord's
  branched-adaptive flexilevel design over a full 0/1 response matrix — N
  (odd) difficulty-sorted items, n = (N+1)/2 administered starting at the
  median (right → easiest harder, wrong → hardest easier), number-right
  self-scoring with +1/2 for a wrong last answer — plus the exact conditional
  score distribution f(x | θ) on the half-integer lattice {1/2, …, n} via
  Lord's forward recursion over p_v(i), taking caller-supplied per-item
  correct-response probabilities (ICC-agnostic). Both primary ETS Research
  Bulletins were READ (ERIC ED042813 / ED051286); the routing is pinned by
  Lord's RWWRWRRRWR worked example and the recursion is cross-checked exactly
  against exhaustive path enumeration. Lord's Eq. 3 efficiency ratio and
  Eq. 4 normal-ogive ICC are deliberately out of scope.
- **Breslow-Day odds-ratio homogeneity DIF test (Breslow & Day, 1980, Eq. 4.30)**
  (`fast_mlsirm.breslow_day_dif`; in Rust `mlsirm_core::dif::breslow_day_dif`,
  PyO3 `py_breslow_day_dif`): the classical NON-UNIFORM DIF companion to
  `mantel_haenszel_dif` — MH tests a common odds ratio against 1; this tests
  whether a common odds ratio is tenable at all across the matching-score
  strata. Per used stratum (all four margins positive) the fitted
  reference-correct count is the admissible root of the fitted-value quadratic
  `A·D/(B·C) = ψ̂` (cancellation-stable q-form roots; defensive
  both-roots admissibility check), the asymptotic variance is
  `1/(1/A + 1/B + 1/C + 1/D)` on the fitted cells, and
  `χ² = Σ (a − A)²/Var` is referred to χ²(K − 1). The plugged-in `ψ̂` is the
  crate's MH `alpha_mh`, the estimator the read source itself endorses
  (worked example: MH 5.158 → χ² 9.28 vs MLE 5.312 → 9.33). Degenerate MH
  odds ratio (`Σad = 0` or `Σbc = 0`), fewer than two usable strata, or an
  inadmissible fitted root yield NaN statistics; Benjamini-Hochberg flags are
  computed across items on the finite p-values. The Tarone (1985) correction
  and the Eq. 4.31 trend test are deliberately out of scope (sources not
  read/documented in the citation-governance header). 0/1 responses only.
- **Generalized Mantel-Haenszel nominal DIF (Zwick, Donoghue & Grima, Eq. 10)**
  (`fast_mlsirm.gmh_dif`; in Rust `mlsirm_core::dif::gmh_dif`, PyO3
  `py_gmh_dif`): unordered-category DIF screening — examinees matched on the
  full total score; within each usable stratum the reference group's
  category-count vector over `T − 1` categories is compared with its
  conditional expectation and covariance; the pooled quadratic form
  `d′S⁻¹d` is referred to χ²(`T_eff − 1`). Effective categories are counted
  in used strata only (categories seen solely in excluded strata do not
  inflate `df`); singular pooled covariance yields NaN (no silent rank
  reduction); category cap `T_eff ≤ 64`. For 0/1 items the statistic equals
  the `mantel_smd_dif` χ² (MH without continuity correction). Integer
  non-negative category codes only (reduced scope; no missing-data support).
- **Mantel polytomous DIF + standardized mean difference (Zwick, Donoghue & Grima)**
  (`fast_mlsirm.mantel_smd_dif`; in Rust `mlsirm_core::dif::mantel_smd_dif`, PyO3
  `py_mantel_smd_dif`): ordinal-item DIF screening — examinees matched on the
  full total score, per-stratum focal score sums compared with their
  conditional hypergeometric expectation/variance (Mantel χ², df = 1, Eqs. 8–9
  of ETS RR-93-14), plus the standardized mean difference effect size (Eq. 11,
  focal-weighted focal-minus-reference item mean difference; weights
  renormalized over usable strata — documented deviation matching the crate's
  standardized P-DIF convention). For 0/1 items the χ² reduces to the MH
  chi-square without continuity correction. Integer non-negative scores only
  (reduced scope; no missing-data support).
- **Empirical Bayes Mantel-Haenszel DIF (Zwick & Thayer)**
  (`fast_mlsirm.eb_mh_dif`; in Rust `mlsirm_core::dif::eb_mh_dif`, PyO3
  `py_eb_mh_dif`): shrinkage enhancement of MH D-DIF statistics — prior
  `N(μ, τ²)` estimated from the supplied item set (`μ` = mean, `τ²` =
  across-item variance minus mean squared SE, floored at 0), per-item
  posterior mean `W·MH + (1−W)·μ` and variance `W·SE²` with
  `W = τ²/(τ² + SE²)`, plus posterior probabilities of the five ETS DIF
  categories (`C−, B−, A, B+, C+`, normal areas delimited at ±1.5/±1).
  Formulas trace to the READ report Zwick & Thayer (2003, LSAC RR / ERIC
  ED481063, statistical-model section); the variance divisor (`n−1`) and
  the degenerate `τ² = 0` point-mass boundary conventions are documented
  implementation choices not printed in the source. Takes MH D-DIF/SE
  pairs (e.g. from `mantel_haenszel_dif`), so any MH pipeline output can
  be stabilized for small samples.
- **Angoff Delta plot DIF detection (deltaPlotR-faithful, response input)**
  (`fast_mlsirm.delta_plot`; in Rust `mlsirm_core::dif::delta_plot`, PyO3
  `py_delta_plot`): transformed item difficulties `4·qnorm(1−p)+13`, the
  R-compatible major axis with `max(b1, b2)` root selection (kept even
  under negative delta covariance, regression-tested), perpendicular
  distances, normal-approximation or fixed detection thresholds, extreme
  proportion handling (`constraint` clamp or `add` correction), and IPP1/
  IPP2/IPP3 iterative item purification with R's membership-row
  convergence semantics — ported from the CRAN deltaPlotR R package's
  `deltaPlot.R` and `adjustExtreme.R` (READ at commit e2aeeb6; Angoff &
  Ford 1973 and Magis & Facon 2012/2014 are cited only as implemented).
  Response-type input only (the R proportion/delta paths, printing, and
  plotting are out of scope); non-{0,1,NaN} responses are rejected rather
  than silently averaged, and returned item indices are 0-based.

- **Nonparametric person-fit statistics (PerFit-faithful, complete data)**
  (`fast_mlsirm.person_fit_np`; in Rust
  `mlsirm_core::personfit_np::person_fit_np`, PyO3 `py_person_fit_np`):
  seven dichotomous statistics — Guttman error count G, normed Guttman
  errors, the norm conformity index NCI, van der Flier's U3 and
  standardized ZU3, Sato's caution index C, and the modified caution
  index C* — ported from the CRAN PerFit R package's `G.R`, `Gnormed.R`,
  `NCI.R`, `U3.R`, `ZU3.R`, `C.Sato.R`, and `Cstar.R` (READ at commit
  c9df433; the originating papers are cited only as implemented).
  Complete 0/1 data only: PerFit's missing-value imputation and
  polytomous variants are out of scope, and any non-{0,1} entry is
  rejected. Perfect (all-0s/all-1s) rows are source-faithful: G, normed
  G, and NCI are 0 (the R source applies `1 - 2*Gnormed` before its
  NaN→0 replacement) while U3/ZU3/C/C* are NaN, and degenerate
  all-equal-difficulty data yields NaN rather than an error. Column
  ordering reproduces R `order(pi, decreasing = TRUE)` including its
  ascending-index tie-break, pinned by a dedicated tie fixture.

- **Hofstee compromise standard setting (psychometricsGP-faithful)**
  (`fast_mlsirm.hofstee`; in Rust
  `mlsirm_core::standard_setting::hofstee`, PyO3 `py_hofstee`): the
  Hofstee compromise cut score, a computational port of the
  psychometricsGP R package's `fn_plot_hofstee()` (`R/fn_plot_hofstee.R`,
  READ — the only inspectable implementation found; single-source port,
  stated openly; plotting excluded; Hofstee 1983 itself NOT READ, cited
  only as implemented). Intersects the piecewise-linear cumulative
  relative frequency ogive over integer score bins 0..=100 (right-closed
  bins `(s-1, s]`, divide-first `(count/n)*100` arithmetic preserved)
  with the descending diagonal `(min_cut, max_fail)` → `(max_cut,
  min_fail)`; when they do not cross, the R fallback pins the cut to
  `min_cut`/`max_cut` with a strict `<` fail count and two-decimal
  DIRECTED rounding (ceil up-branch / floor down-branch), `failed=True`.
  Reduced scope per adversarial spec review: collinear ogive-diagonal
  overlap and zero-length diagonals are rejected (`spatstat`
  `crossing.psp` degenerate semantics unverified against an R runtime).
- **K1/K2/S1/S2 answer-copying indices (CopyDetect-faithful)**
  (`fast_mlsirm.k_variants`; in Rust `mlsirm_core::security::k_variants`,
  PyO3 `py_k_variants`): the four regression-baseline copying indices,
  ported exactly from the CRAN CopyDetect package's internal `ks12()`
  (`R/similarity1.r`, READ), specialized to complete scored 0/1 data (no
  missing responses — the port rejects anything but exact 0/1). Number-
  incorrect subgroups EXCLUDE the source (the opposite of `k_index`'s base
  `k()` convention — a deliberate CopyDetect asymmetry, regression-anchored
  in tests). K1/K2 fit linear/quadratic least squares of subgroup incorrect-
  match rates and take binomial upper tails `P(Bin(ws, p) >= m)`; S1/S2 fit
  log-linear Poisson GLMs of (weighted) match counts and take bounded
  Poisson WINDOW probabilities (`P(m <= X <= ws)` / `P(mm <= X <= n_items)`,
  not plain upper tails — CopyDetect subtracts the tail beyond the cap),
  with S2 adding the `(1.5e)^(-6·prob)` weighted correct-match term
  and a RAW ceiling (`mm = ceil(sum) + m`, no epsilon — float noise at
  integer boundaries can bump `mm`, documented). Numerics: rank-checked
  modified Gram-Schmidt QR for the OLS fits (degenerate designs raise, no
  silent normal-equation blowup) and a guarded Newton Poisson GLM with
  step-halving, bounded eta, and a stable start (nonconvergence raises);
  `ks12()` itself SUPPRESSES R's non-integer-Poisson warning for the S2
  fit. Sotaridona & Meijer (2002, *JEM 39*(2)) and (2003, *JEM 40*(1)) NOT
  read — all four indices cited only as implemented by CopyDetect.
- **Generalized binomial test (GBT) tail kernel (aberrance-faithful)**
  (`fast_mlsirm.gbt`; in Rust `mlsirm_core::security::gbt`, PyO3 `py_gbt`):
  exact Poisson-binomial distribution of the copier-source match count via
  Bernoulli-convolution DP and the INCLUSIVE upper-tail p-value
  `P(M >= observed)`, ported exactly from the CRAN aberrance package's
  `compute_GBT` (`src/compute.cpp`, READ) and corroborated by CopyDetect's
  internal `GBT()` (`R/similarity1.r`, READ — same distribution, same
  inclusive tail). Per-item match-probability construction is the caller's
  job (aberrance directional and CopyDetect symmetric recipes both fit);
  missing data out of scope (the packages conflict). van der Linden &
  Sotaridona (2006) NOT read — cited only as implemented. Returns the full
  pmf plus the p-value; O(n^2) time / O(n) memory nonnegative f64 DP — no
  cancellation, but tiny extreme large-n masses may underflow.
- **K-index of matching incorrect answers (CopyDetect-faithful)**
  (`fast_mlsirm.k_index`; in Rust `mlsirm_core::security::k_index`, PyO3
  `py_k_index`): binomial upper-tail index of copier-source shared incorrect
  answers against a number-incorrect subgroup baseline, ported exactly from
  the CRAN CopyDetect package's internal `k()` (`R/similarity1.r`, READ;
  corroborated by `R/similarity2.r`), with the binomial tail summed in log
  space (no factorial overflow or extreme-p underflow). The subgroup
  includes the copier and,
  when scores match, the source (CopyDetect convention). Holland (1996,
  RR-96-07) and Sotaridona & Meijer (2002) NOT read — cited only as
  implemented; Sotaridona & Meijer (2001, ERIC ED467373) read for
  background. Validation rejects non-binary/complex/bool inputs and the
  degenerate all-correct source.
- **Omega answer-copying statistic (Wollack-style)**
  (`fast_mlsirm.wollack_omega`; in Rust `mlsirm_core::security::wollack_omega`,
  PyO3 `py_wollack_omega`): standardized index of answer similarity between a
  suspected copier and a source. `h` counts identical observed options,
  `p_i = P_i[source_i]` is the copier's model-implied probability of the
  source's observed option, `omega = (h - sum p_i)/sqrt(sum p_i (1 - p_i))`
  with a one-sided upper-tail normal p-value. Formula verified against two
  independently READ implementations: the CRAN CopyDetect R sources
  (`similarity1.r`/`similarity2.r`) and the aberrance package
  (`compute_OMG`); NOT read: Wollack (1997, *Applied Psychological
  Measurement, 21*(4), 307-320) itself (access blocked) — cited only as
  implemented by those sources. CopyDetect's printed docs flip the sign
  (`(E-h)/sqrt(V)`) but both source files use `(h-E)/sqrt(V)`; the source
  convention is implemented. Scope: omega only — no g2/GBT/K-index, no
  continuity correction, no missing responses; the caller supplies the
  copier's fitted option probabilities (e.g. from a nominal response model).
  Pinned against an independent Python oracle at 1e-12 (p-values 5e-7 via
  crate erfc); error paths, structural single-item-extension invariant, and
  a 500-rep Monte Carlo size/power check (`#[ignore]`); 3 executed mutation
  kills (V-vs-sqrt(V) scaling, copier-probability lookup, two-sided p).

- **DIMTEST test of essential unidimensionality (original Stout-style
  AT1/AT2 statistic)** (`fast_mlsirm.dimtest`; in Rust
  `mlsirm_core::detect::dimtest`, PyO3 `py_dimtest`): confirmatory
  hypothesis test with caller-supplied assessment subtests AT1/AT2 (equal
  length >= 4, disjoint) and the complementary partitioning subtest PT;
  examinees are grouped by raw PT total score (groups smaller than 20
  discarded), within each retained group the observed ML variance of AT
  totals is compared to the local-independence variance
  `sum_i p_i (1 - p_i)` normalized by Stout's standard-error estimate
  `S_k`, giving `T_L = K^{-1/2} sum_k (sigma_k^2 - sigma_U,k^2)/S_k`, the
  AT2 bias correction `T_B`, and `T = (T_L - T_B)/sqrt(2)` with a one-sided
  upper-tail normal p-value. Formulas transcribed from Nandakumar & Stout's
  1992 ERIC technical report ED351383 (published 1993, *Journal of
  Educational Statistics, 18*(1), 41-68), which describes Stout (1987,
  Sec. 4); Kieftenbeld & Nandakumar (2015, PMC5978610) READ for the
  original-vs-bootstrap bias-correction distinction. NOT read: Stout (1987)
  original article, Stout et al. (2001), Froelich & Habing (2008), DIM-Pack
  sources — no ATFIND, no DIMTEST 2 / bootstrap correction, no polytomous
  items, no missing data. Pinned against an independent NumPy oracle
  (500x18 two-dimensional fixture, agreement 1e-12 on `T_L`/`T_B`/`T`;
  p-value at 5e-7 due to the crate's Numerical Recipes `erfc`).

- **Confidence-interval (ACI) classification for CAT**
  (`fast_mlsirm.ci_classify`; in Rust `mlsirm_core::exposure::ci_classify`,
  PyO3 `py_ci_classify`): single-cut binary-response classification by
  interim EAP ability estimate on a fixed 41-point `[-4, 4]` grid with
  standard-normal prior, SE = EAP posterior SD, interval
  `theta_hat +/- z_crit * se` vs `theta_cut` with STRICT first-crossing
  decisions -> `"above"`/`"below"`/`"continue"` with 1-based `n_used`; full
  theta/se/lower/upper traces are returned as offline diagnostics (entries
  past `n_used` are counterfactual replay values). Verified against R catIrt
  `termCI.R`/`eapEst.R`/`catIrt.Rd` at commit
  `c9e979e4812c27d95d367a7f097edfe8e93ac8eb` (READ); Kingsbury & Weiss
  (1983), Thompson (2007), and Eggen & Straetmans (2000) were NOT
  method-section verified and are historical/background context only.
- **Wald SPRT classification for CAT** (`fast_mlsirm.sprt_classify`; in
  `mlsirm_core::exposure`). Single-cut binary-response sequential probability
  ratio test: point hypotheses at `theta_cut -/+ delta`, cumulative binary
  log-likelihood ratio under the D=1 logistic 3PL, and inclusive
  first-crossing decisions against the log Wald boundaries
  `A = ln((1-beta)/alpha)`, `B = ln(beta/(1-alpha))` -> `"above"`/`"below"`/
  `"continue"` with 1-based `n_used`; the full `llr_trace` is returned as an
  offline diagnostic (entries past `n_used` are counterfactual replay
  values). Verified against R catIrt `termSPRT.R`/`logLik.brm.R`/`p.brm.R`
  and Thompson (2007, doi:10.7275/fq3r-zz60); Reckase (1983), Eggen (1999),
  and Wald (1947) are cited as historical origins via Thompson (not directly
  read). Log-likelihood ratios are computed in stable log space (softplus /
  log-sigmoid), so extreme-but-valid parameters that saturate the response
  probability to numerical 0/1 yield finite LLRs instead of errors. Pinned
  17-digit interior-crossing oracle, error-path and 500-rep Monte-Carlo
  structural-invariant tests; 4 executed mutation kills (swapped boundaries,
  dropped guessing floor, collapsed null hypothesis, off-by-one `n_used`).
- **Owen-approximate posterior-predictive EPV item selection**
  (`fast_mlsirm.epv_select`; in `mlsirm_core::exposure`). Deliberately
  reduced scope of van der Linden's (1998, doi:10.1007/BF02294775) minimum
  expected posterior variance (MEPV) criterion: the posterior is Owen's
  normal approximation `N(mu, sig2)`, the predictive probability is
  `p*_i = c_i + (1-c_i) Phi((mu-b_i)/sqrt(1/a_i^2 + sig2))`, and the outcome
  posterior variances come from `owen_update` rather than exact numerical
  posteriors; the unadministered item minimizing
  `EPV_i = p*_i sig2_i^+ + (1-p*_i) sig2_i^-` is selected (lowest-index
  ties). van der Linden (1998) READ as ERIC ED424235 (Research Report
  96-01); the exact-MEPV contract additionally verified against R catR
  `EPV.R` and mirtCAT `selection_criteria.R` (both READ); Owen (1975) NOT
  read (update formulas follow the crate's `owen_update`). Pinned oracles
  and a delegation discriminator (argmin EPV vs. max-info vs. b-matching)
  fixed by the adversarial spec review.
- **Kingsbury-Zara constrained CAT (CCAT) content balancing**
  (`fast_mlsirm.ccat_select`; in `mlsirm_core::exposure`). Single-step
  content-balanced item selection: eligible groups with zero administered
  items have priority, otherwise the eligible group with the maximal
  target-minus-empirical-proportion discrepancy is chosen; within the chosen
  group the unadministered item with maximal logistic 3PL Fisher information
  `a^2 (Q/P) ((P-c)/(1-c))^2` is selected. Ties go to the lowest index
  (documented deterministic deviation from catR's random tie-break).
  Kingsbury & Zara (1989, doi:10.1207/s15324818ame0204_6) itself NOT read
  (paywalled); the rule is implemented as reproduced by the R catR package
  (`nextItem.R` `cbControl` branch; READ), and the information formula was
  verified against catR `Ii.R`/`Pi.R`. Pinned oracles computed in exact
  arithmetic by the adversarial spec review.
- **Owen approximate Bayesian sequential CAT** (`fast_mlsirm.owen_update`,
  `fast_mlsirm.owen_cat`; in `mlsirm_core::exposure`). Closed-form
  normal-approximation posterior moment updates for the 3PNO model
  (`P = c + (1-c)Phi(a(theta-b))`) and a sequential driver with Owen's
  b-matching selection (`argmin |b_i - mu|`, ties to the lowest index) and
  posterior-variance stopping rule (plus a `test_length` cap). Owen (1975)
  itself NOT read (paywalled); formulas implemented as reproduced by
  van der Linden (1998, Research Report 96-01, Appendix A.1-A.6) and
  cross-checked against the R `irt` package `est_ability_owen.cpp`; pinned
  oracles verified against exact-posterior numerical integration (~1e-13)
  by the adversarial spec review.

- **Chang-Ying KL global-information CAT selection**
  (`fast_mlsirm.kl_information`, `fast_mlsirm.kl_select`; in
  `mlsirm_core::exposure`). Kullback-Leibler item index as the UNNORMALIZED
  area of the pointwise Bernoulli divergence (expectation under the
  provisional `theta0`) over `[theta0 - delta, theta0 + delta]` via composite
  Simpson (2048 panels), and next-item selection with the paper's shrinking
  half-width `delta = r / sqrt(n_administered)` (requires `n >= 1`; `r = 3`
  default per Study 1). Administered items keep their computed index; masking
  applies to selection only. Small-delta Fisher limit
  `I(theta0) * delta^3 / 3` anchored by test. Paper READ (Chang & Ying, 1996,
  doi:10.1177/014662169602000303); cross-checked against catR `KL.R`.

- **Raju ICC-area DIF** (`fast_mlsirm.raju_area`; in `mlsirm_core::dif`).
  Parametric signed/unsigned area between two logistic ICCs on a common
  scale, with Raju's delta-method Z tests. Signed area `h = b_F - b_R`
  (positive = harder for focal); unsigned `h = |H|` from Raju's closed form
  via a numerically stable softplus, with a continuous equal-slope fallback
  `|b_F - b_R|`; common-guessing 3PL reports `h`/`se` scaled by `(1 - c)`
  with the Z from unscaled quantities. Primary papers NOT read (Raju 1988,
  Psychometrika 53(4), 495-502, doi:10.1007/BF02294403; Raju 1990, APM
  14(2), 197-207, doi:10.1177/014662169001400208 — both paywalled); formula
  oracle is the difR source (`RajuZ.R`, `difRaju.R`; Magis et al. 2010,
  Behavior Research Methods 42, 847-862, doi:10.3758/BRM.42.3.847 — code
  read in full, package paper not read). Both areas and all four
  delta-method partials re-derived by hand and verified against numeric
  quadrature/finite differences in adversarial spec review; documented difR
  divergences: its gradient is the uniform negation of dH (variance-
  equivalent) and its `exp(Y)==Inf` overflow branch carries the sign
  opposite to the closed-form positive-side limit (unreachable here via
  softplus + `|H|`). Monte-Carlo (500 reps, parametric asymptotic):
  signed test holds nominal level and power; the unsigned Z is measurably
  anti-conservative under an exact equal-slope null (~.14 at nominal .05),
  documented in the API rather than hidden.

- **Velicer minimum average partial (MAP) test** (`fast_mlsirm.velicer_map`,
  `velicer_map_from_data`; in `mlsirm_core::factor`). Component-retention
  test of Velicer (1976, Psychometrika 41(3), 321-327,
  doi:10.1007/BF02293557 — NOT read; formula
  support is the read implementations below): PCA loadings from the
  eigendecomposition of R, partial covariance `C* = R - A_m A_m'` rescaled
  to a partial correlation matrix, and `f2[m]` = mean squared off-diagonal
  partial correlation for `m = 0..max_m` (with the `m = 0` baseline being
  R itself); retained components = the `m` at the minimum. Also computes
  the revised elementwise fourth-power criterion `f4` (Velicer, Eaton, &
  Fava, 2000, in Goffin & Helmes, Problems and Solutions in Human
  Assessment, 41-71 — not read; attributed per O'Connor's code comments).
  Algorithm and retention rule verified against Brian O'Connor's canonical
  MAP programs (map.m and map.sps, oconnor-psych.ok.ubc.ca/nfactors — read
  in full; O'Connor, 2000, Behavior Research Methods, Instruments, &
  Computers 32(3), 396-402, paper itself not read) and psych VSS.R `map()`
  (Revelle, 2025 — read). Documented divergences found by adversarial
  review: `fungible::faMAP` prints a 1-based row position (off by one vs
  O'Connor's count — not reproduced); `EFA.dimensions::MAP` now uses matrix
  powers for the fourth-power criterion, conflicting with O'Connor's
  elementwise form (unresolved from primary literature; we follow
  O'Connor). Rows with singular partial-covariance normalization (e.g.
  identity R for `m >= 1`) are NaN and excluded from the argmin. Rust core
  with PyO3 binding and thin NumPy wrapper; Harman-8 full-vector oracle
  parity (independent NumPy transcription), identity guard, and a
  500-replication Monte-Carlo recovery test (`#[ignore]`).
- **a-stratified multistage CAT item selection** (`fast_mlsirm.a_stratified`;
  in `mlsirm_core::exposure`). Simulation of Chang & Ying's (1999)
  a-stratified design: the pool is split into `n_strata` contiguous strata by
  ascending discrimination `a` (stable sort, near-equal sizes with the first
  `n mod K` strata one item larger — repository choice; catR places the
  remainder last), the test is partitioned into matching stages, and within
  the active stratum the next item is `argmin |b_i - theta_hat|`
  (b-matching, ties to the lowest original index). The b-matching selection
  rule and ascending-a strata are confirmed from Barrada, Mazuela, & Olea
  (2006, Psicothema 18(1), 156-159 — read in full); Chang & Ying (1999,
  Applied Psychological Measurement 23(3), 211-222) is cited as the design's
  origin from its abstract. Interim EAP on a uniform grid and the initial
  `theta_hat = 0` are repository choices (the paper used ML-based interim
  estimation). Returns per-item exposure rates, stratum assignment, stage
  lengths, and theta RMSE/bias; the per-stratum counting identity
  `sum_{i in stratum k} P(A_i) = stage_lengths[k]` holds exactly and is
  regression-tested against returned values. Stratum-level b-blocking
  (Chang, Qian, & Ying, 2001, "a-stratified multistage computerized adaptive
  testing with b blocking", Applied Psychological Measurement 25(4),
  333-341 — not read; excluded per adversarial spec review) is out of
  scope. Rust core with PyO3 binding
  and thin NumPy wrapper; mutation-audited tests plus a 500-replication
  Monte-Carlo comparison (`#[ignore]`) showing lower exposure imbalance
  (summed squared deviation from the uniform rate `L/n`) than
  max-information selection.
- **Sympson-Hetter item-exposure control** (`fast_mlsirm.sympson_hetter`;
  in `mlsirm_core::exposure`). Iterative Monte-Carlo calibration of the
  exposure-control parameters `k_i = P(A_i | S_i)` for dichotomous 3PL
  max-information CAT with interim EAP: per-encounter uniform gate
  (administer iff `u <= k_i`, rejected items blocked for the remainder of
  that simulee's test), update `k_i <- min(1, r_max / P(S_i))` (Barrada,
  Olea, & Ponsoda, 2007, Eq. 1-3; algorithm confirmed from Georgiadou,
  Triantafillou, & Economides, 2007 — both read in full; Sympson & Hetter,
  1985, cited as origin, not read). The stopping rule
  `max P(A) <= r_max + tol` is a practical criterion, not a convergence
  theorem (van der Linden, 2003, abstract); the returned `k` is always the
  vector that produced the reported final-cycle rates. Feasibility bound
  `r_max >= test_length/n_items` (exact counting identity
  `sum_i P(A_i) = test_length`, derived here) is enforced; the bound is
  necessary, not sufficient — a tight `r_max` near the bound may still
  exhaust the pool mid-test and fail with the documented error. `r_max = 1`
  reduces exactly to unconstrained max-info CAT (no exposure RNG
  consumed); an exhausted pool raises an error (repository policy, not a
  classical prescription). Adversarially spec-verified before
  implementation (REDUCED SCOPE: no theta-stratified variants, no
  forced-administration fallback, no "classical iteration count" claim);
  a 500-rep Monte Carlo calibration run (`#[ignore]`); four executed
  mutation kills (gate flip, `P(A)` update denominator, no-blocking —
  killed by divergence/non-termination — and swapped selection/exposure
  bookkeeping).
- **Selection utility analysis** (`fast_mlsirm.selection_utility` /
  `taylor_russell`; in `mlsirm_core::utility`; transcribed from CRAN
  iopsych 0.90.1 `utilityBcg`/`trModel`/`ux`, read in full — Goebl,
  Jones, & Beatty, 2016; the original Taylor & Russell, 1939, Naylor & Shine, 1965,
  and Cronbach & Gleser, 1965 sources were not read and are cited as
  attributed). Formulas under the standard bivariate-normal selection
  model: selection intensity `ux = phi(xc)/sr`, Naylor-Shine selected-group
  criterion mean `pux = rxy*ux`, BCG utility gain
  `n*period*sdy*pux - cost_total` (the iopsych `cost` argument is
  documented here as a TOTAL cost — iopsych labels it per-applicant but
  never multiplies by `n`), and Taylor-Russell success ratio
  `P(Y>yc | X>xc) = Q(xc,yc,rxy)/sr` with the bivariate-normal upper tail
  `Q` evaluated by a conditional-normal Gauss-Legendre integral (~1e-15
  vs scipy's BVN CDF at moderate `|rxy|` during adversarial spec review,
  better than 1e-6 across the whole accepted `|rxy|` range —
  regression-tested; the committed oracle generator is
  `tests/oracles/oracle_utility.py`; the iopsych `qa/(qa+qb)` form was
  proven equal to `Q/sr` algebraically and numerically). Adversarially spec-verified before implementation; five
  scipy-pinned oracle fixtures including negative validity; rho=0
  analytic anchor (success == base rate); strict rxy monotonicity; a
  500-rep x 20,000-person Monte Carlo recovery run (`#[ignore]`; success
  ratio within 4.3e-5, pux within 3.8e-4); four executed mutation kills
  (dropped `rxy` in pux, sign flip in the Q integrand, `1-sr` denominator
  in ux, sr/br role swap). Documented identity limitation: the mutant
  `Q(h,k) -> Q(k,h)` alone is output-identical everywhere by BVN exchange
  symmetry — no test claims to kill it; cutoff-role bugs are anchored by
  the role-swap kill instead. Post-implementation adversarial review
  hardening: sub-ulp ratios (where `1.0 - v` rounds to 1.0) are rejected
  instead of returning NaN/silent zeros; the BVN panel width scales with
  `sqrt(1 - rho^2)` so `|rxy|` near 1 stays accurate (Err beyond
  `sqrt(1-rxy^2) < 1e-4`); `q_joint` is bounded by `min(sr, br)`; all
  three regression-tested against scipy `quad` oracles.
- **Factor-analytic greatest lower bound** (`fast_mlsirm.glb_fa` /
  `glb_fa_from_data`; in `mlsirm_core::factor::glb_fa_corr`; transcribed
  from CRAN psych `glbs.R` `glb.fa`, read in full — Revelle, 2025; NOT the
  algebraic glb of `glb.algebraic`, which requires an SDP solver; Sijtsma,
  2009, not read). Algorithm: 1-factor minres fit, eigenvalues of `R` with
  the diagonal replaced by the model communalities, `nf` = count of
  positive eigenvalues with psych's single df-based decrement, then
  `glb = sum(rr)/sum(R)` with `diag(rr)` from an `nf`-factor refit.
  Verified against a pinned independent scipy oracle on a 9-variable
  2-factor population matrix (glb to 1e-5), a sampled 6-variable matrix
  and a df-adjustment fixture (both df = 0 saturated fits, wider bands
  documented), plus a 500-rep Monte Carlo run (`#[ignore]`; observed mean
  glb 0.863 vs population omega 0.830 — the expected upward bias of glb
  under multi-factor detection is documented, not hidden). Three executed
  mutation kills (skipped diagonal substitution, 1-factor communalities in
  the ratio, dropped df decrement).
- **Person separation reliability** (`fast_mlsirm.separation_reliability`;
  in `mlsirm_core::reliability`; transcribed from CRAN eRm `SepRel.R`, read
  in full — Mair et al., 2025; the statistic is attributed there to Wright
  & Stone, 1999, not read). `R = (SSD - MSE)/SSD` with `SSD = var(measures)`
  (n-1 denominator) and `MSE = mean(se^2)`, unclamped; plus the hand-derived
  separation index `G = sqrt((SSD - MSE)/MSE)` (adjusted true SD over RMSE,
  `G^2 = R/(1-R)`; not in the read source). eRm's extreme-score/NA
  filtering is documented as caller responsibility. Verified against a
  pinned numpy fixture (SSD, MSE, R, G at 12 decimals), a negative-R path,
  and a 500-rep Monte Carlo recovery (`#[ignore]`; population R = 0.8
  recovered to 0.01); three executed mutation kills (swapped numerator,
  population variance, unsquared se).
- **Minres (ULS) exploratory factor analysis and McDonald's omega_total
  (1-factor)** (`fast_mlsirm.minres_fa`, `minres_fa_from_data`,
  `omega_total_1f`, `omega_total_1f_from_data`; in `mlsirm_core::factor`;
  line-by-line transcription of CRAN psych `fa.R`'s minres path — Revelle,
  2025, read; McDonald, 1999, cited-not-read, the omega formula is
  hand-derived from the standardized 1-factor model). Uniquenesses are
  box-constrained to `[0.005, 1]` and optimized by projected
  Barzilai-Borwein descent with an Armijo safeguard and finite-difference
  fallback (psych's `FAgr.minres` direction is not the exact gradient of
  the lower-triangle objective — a verified limitation); convergence is
  certified by a finite-difference box-KKT check whose maximum violation
  is returned (`kkt_violation`). Loadings are unrotated, columns in
  descending-eigenvalue order with column sums >= 0. REDUCED SCOPE: no
  rotation, no Schmid-Leiman / omega_hierarchical, no ML/WLS/GLS, no
  factor scores. Tests pin parity at 5e-5 against an independent scipy
  L-BFGS-B transcription oracle, anchor the absolute objective value of a
  deliberately misfitting 1-factor fit (the only assert that can kill the
  lower-triangle-vs-all-off-diagonal x2 mutation — disclosed), verify
  rank-1 exact recovery and structure invariants, execute four mutation
  kills, and include a 500-rep `#[ignore]` Monte Carlo recovery study.
- **Generalizability theory G/D studies for crossed designs**
  (`fast_mlsirm.gtheory_pi`, `fast_mlsirm.gtheory_pio`; in
  `mlsirm_core::gtheory`; Huebner & Lucht, 2019, read in full — the EMS
  inversions the paper defers to Brennan, 2001, and Shavelson & Webb,
  1991, both cited-not-read, are hand-derived and numerically verified
  against the paper's published Tables 3-6). One-facet `p x i` and
  two-facet `p x i x o` random-effects ANOVA variance components, plus
  D-study relative/absolute error variances, generalizability coefficient
  E-rho^2, and dependability Phi over proposed facet sizes. Negative raw
  components are reported as-is in `var_raw` and clamped to zero in `var`
  for the D study (documented clamped-ANOVA implementation policy);
  coefficients are NaN when their denominator is <= 1e-12. Rust tests
  reproduce the paper's worked examples at full precision, add
  independent RNG-pinned fixtures (including a natural negative-component
  anchor), executed mutation kills (M1/M3/M5/M6), and a 500-rep
  `#[ignore]` Monte-Carlo recovery test.
- **Livingston-Lewis classification accuracy and consistency**
  (`fast_mlsirm.livingston_lewis`; in `mlsirm_core::classification`;
  Livingston & Lewis, 1995, as implemented in Haakstad's CRAN
  `betafunctions` 1.9.0 source `LL.CA` in `R/classification.R`, read line
  by line — the original article was not consulted directly; Hanson, 1991,
  four-parameter beta moment fit, as cited in Haakstad, 2022). From a
  single administration: effective test length
  `((m-min)(max-m) - r s^2)/(s^2 (1-r))`, true-score raw moments via the
  factorial-moment identity on the unrounded-ETL scale, four-parameter
  beta method-of-moments fit with a two-parameter [0, 1] fail-safe, then
  accuracy cells (tp/fp/tf/ff), sensitivity/specificity, consistency
  cells, and Cohen's kappa under a binomial observed-score model with
  `N = round(ETL)`. Integrals use singularity-safe composite
  Gauss-Legendre quadrature (power substitution when a shape parameter is
  below one; endpoint-graded panels otherwise), verified against
  `scipy.integrate.quad` replication literals at 1e-7. Divergences
  (documented in the module): a single round-ties-even threshold
  `k = round(N c)` in both the accuracy and consistency blocks (the oracle
  mixes `round` in accuracy with `floor` in consistency, making its
  consistency cells asymmetric; here `p_ij == p_ji` by construction);
  pass = observed score >= cut is the positive class (the oracle labels
  fail as positive); the fail-safe also engages on numerically invalid
  four-parameter fits (the oracle only checks out-of-bounds support); hard
  errors instead of NA/NaN propagation for invalid inputs, while the
  conditional ratios (sensitivity, specificity, kappa) are an explicit
  `NaN` when their margin or chance denominator vanishes (e.g. a cut
  outside the fitted beta support).
- **Cronbach alpha + Feldt exact-F confidence interval**
  (`fast_mlsirm.cronbach_alpha`, `fast_mlsirm.feldt_alpha_ci`; in
  `mlsirm_core::reliability`; Feldt, 1965, as cited in and implemented by
  Revelle's CRAN `psych` 2.6.5 source `alpha.ci` in `R/alpha.R`, read line
  by line; Cronbach, 1951, covariance form verified against the same
  source). `cronbach_alpha` computes the raw-covariance form
  `p/(p-1) * (1 - tr(C)/sum(C))`; `feldt_alpha_ci` inverts the pivot
  `(1-alpha)/(1-alpha_hat) ~ F(n-1, (n-1)(p-1))` into a two-sided interval
  (`lower = 1-(1-alpha_hat)*qF(1-delta/2)`, upper mirrored) plus the
  implied average inter-item correlation `r_bar`. The F quantile is
  computed in-crate via a Lentz continued-fraction regularized incomplete
  beta and bisection (verified against `scipy.stats.f.ppf` fixture
  literals at 1e-9). Bounds are not clamped; negative alpha is accepted
  into the CI, matching psych. Divergences (documented in the module):
  raw-data input only, zero-variance items rejected, confidence `level`
  argument instead of `p.val`, hard errors instead of NA.
- **ten Berge & Zegers mu reliability series** (`fast_mlsirm.tenberge_mu`;
  in `mlsirm_core::reliability`; ten Berge & Zegers, 1978, as cited in and
  implemented by Revelle's CRAN `psych` 2.6.5 source `tenberge.R`, read
  line by line). On the Pearson correlation matrix with `Vt = sum(R)`,
  off-diagonal power sums `S_k`, and `c = p/(p-1)` on the innermost radical
  only: `mu0 = c*S_1/Vt` (= coefficient alpha = Guttman lambda3),
  `mu1 = (S_1 + sqrt(c*S_2))/Vt` (= Guttman lambda2), `mu2` and `mu3` nest
  one and two further radicals over `S_4` and `S_8`. The series ordering
  `mu0 <= mu1 <= mu2 <= mu3` follows from Cauchy-Schwarz over the `p*(p-1)`
  off-diagonal cells and is asserted on crate outputs. Divergences from
  psych (documented in the module): raw-data input only (no
  correlation-matrix passthrough, no `use = "pairwise"`), hard errors on
  degenerate input, `S_1` summed directly to avoid cancellation. Verified
  against an independent NumPy replication on two fixtures pinned at
  `1e-9`, exact-identity cross-checks against `guttman_lambdas`, and a
  500-replication tau-equivalent Monte Carlo (`#[ignore]`).
- **Guttman lambda reliability coefficients** (`fast_mlsirm.guttman_lambdas`;
  new `mlsirm_core::reliability`; Guttman, 1945, as cited in and implemented
  by Revelle's CRAN `psych` 2.6.5 sources `guttman.R`/`splitHalf.R`/`smc.R`,
  read line by line). On the Pearson correlation matrix: lambda1-lambda3
  (lambda3 = coefficient alpha), lambda5 (best covariance column), lambda6
  (squared multiple correlations via a plain symmetric inverse), plus
  split-half summaries — lambda4 (best split), beta (worst split, floored at
  0), and the mean split over all `C(p, floor(p/2))` subsets when that count
  fits the `n_sample_splits` budget (psych's brute-force cutoff 15000),
  otherwise over LCG-sampled splits. Declared divergences (documented in the
  module): no `check.keys` auto-reversal, absolute split-half correlations
  in both branches (psych's sampled branch is signed), hard error on
  singular correlation matrices instead of psych's pseudoinverse, crate-LCG
  sampling (psych-inspired, not bit-identical to any R run), and duplicate
  sampled subsets allowed. Verified against an independent NumPy replication
  (`np.corrcoef` + `np.linalg.inv` + `itertools.combinations`) on three
  fixtures (even-p exhaustive, odd-p exhaustive, sampled) pinned at `1e-9`,
  plus a 500-replication tau-equivalent Monte Carlo (`#[ignore]`) recovering
  the analytic sum-score reliability within 0.01.
- **Horn's parallel analysis** (`fast_mlsirm.parallel_analysis`; new
  `mlsirm_core::parallel`; Horn, 1965, and Glorfeld, 1995, as cited in and
  implemented by Dinno's CRAN `paran` 1.5.6 sources, read line by line).
  PCA path: eigenvalues of the observed Pearson correlation matrix (cyclic
  Jacobi, eigenvalues only, hard error on non-convergence) are adjusted by
  the sampling bias `random_eigenvalue - 1` estimated from `n_iterations`
  standard-normal data sets of the same shape; components are retained
  while the adjusted eigenvalue stays above 1, scanning left to right and
  stopping at the first failure (later resurgences do not count, matching
  paran's loop-and-break). `centile = 0` uses the per-position mean
  benchmark; `1..=99` uses that upper centile via the R type-7 quantile
  (Glorfeld's conservative variant). Deliberate divergences (documented in
  the module): PCA only (paran's `cfa` generalized-inverse path is out of
  scope), a single deterministic crate-LCG random stream (paran-inspired,
  not bit-identical to any R run), and narrowed guards (`n_persons >= 3`,
  `n_items >= 2`, finite complete data, positive column variance, explicit
  `n_iterations`; the Python wrapper supplies paran's `30 * n_items`
  default). Fixture literals verified against an independent NumPy
  replication that mirrors the LCG stream exactly.

- **IRT classification accuracy and consistency**
  (`fast_mlsirm.rudner_classification`, `fast_mlsirm.lee_classification`; new
  `mlsirm_core::classification`; Rudner, 2001, 2005; Lee, 2010, as cited in
  Lathrop's CRAN `cacIRT` sources). Rudner's normal-approximation method
  treats the observed score at ability theta as N(theta, sem^2) and reports
  per-cut and simultaneous accuracy/consistency, conditional and marginal
  (weights normalized internally; uniform weights reproduce cacIRT's
  person-level `Rud.P`, quadrature weights the distribution-level `Rud.D`).
  Lee's summed-score method replaces the normal approximation with the exact
  Lord-Wingersky (1984) score distribution reused from
  `mlsirm_core::scoring`; raw cuts split scores at `ceil(cut)` and the true
  category is the raw-score interval containing the expected true score.
  Category intervals are left-closed everywhere (cacIRT's `Lee.D` alone is
  right-closed — documented divergence); item probabilities must lie
  strictly inside (0, 1) (rejecting P == 0 is stricter than the oracle,
  which only breaks at P == 1); simultaneous outputs are always populated
  (cacIRT emits them only for two or more cuts). Polytomous Lee, `np.cac`,
  and the MLE/SEM ability helpers are out of scope. Rudner outputs inherit
  the crate `erfc` accuracy (|err| < 1.2e-7); Lee outputs are exact to f64
  rounding. For LLM-as-a-Judge quality management this quantifies how
  reliably a judge's cut score separates pass from fail. Rust-only numerics;
  the Python wrapper validates and marshals. Tests pin marginals and
  conditionals against literals from an independent NumPy transcription
  (exact `math.erf`, own recursion), anchor left-closed cuts with a theta
  exactly on a cut and a dyadic true score exactly on a raw cut, use
  unnormalized weights and a non-integer raw cut as mutation anchors (four
  mutation spot-checks killed), and include a 500-replication ignored Monte
  Carlo ordering long informative tests above short noisy ones.
- **Confirmatory DETECT dimensionality analysis** (`fast_mlsirm.detect_analysis`;
  new `mlsirm_core::detect`; Zhang & Stout, 1999, as cited in Robitzsch, 2024).
  Estimates pairwise conditional covariances of binary items with sum-score
  conditioning — the bias-corrected average of the total-score and pair
  rest-score conditionings, per-group ML covariance aggregated with
  group-frequency weights — and computes the DETECT, ASSI, RATIO, MADCOV100,
  and MCOV100 indices against a known item clustering (labels opaque,
  equality-only). Transcribed line-by-line from the CRAN `sirt` R sources
  (`detect.index.R`, `ccov.np.R`, `ccov_np_compute_ccov_sum_score.R`,
  `conf.detect.R`); matches the explicit `ccov.np(use_sum_score=TRUE,
  scale_score=FALSE)` path — the kernel-smoothed default, missing data
  (sirt pairwise-deletes), sqrt(N)-weighted variants (coincide with
  unweighted under complete data), exploratory cluster search, and polytomous
  DETECT are documented as out of scope. All-zero conditional covariances
  (RATIO `0/0`, NaN in R) are rejected with an error. For LLM-as-a-Judge
  item-quality management this diagnoses whether a rubric partition of judge
  items behaves as distinct dimensions. Rust-only numerics; the Python
  wrapper validates and marshals. Tests pin all five indices and every
  per-pair conditional covariance against literals from an independent NumPy
  transcription (which cannot discriminate the z-standardized default path,
  since unique-value grouping is invariant to monotone transforms — the scope
  statement pins that contract), plus hostile `i64::MIN`/`i64::MAX` labels
  and a 500-replication Monte Carlo (`#[ignore]`) separating 2D simple
  structure from unidimensional data.
- **Haberman subscore added-value analysis** (`fast_mlsirm.subscore_analysis`;
  new `mlsirm_core::subscores`; Haberman, 2008, as cited in Sinharay, 2010).
  For each subscale of a disjoint, exhaustive item partition computes the
  PRMSEs of the three classical-test-theory true-subscore estimators — from
  the observed subscore (`= Cronbach alpha`), from the observed total
  (`rho^2(s_t, x_t) * alpha_x` with the true-score covariance row sum over
  subscore columns only), and from both jointly (Wainer-style augmentation via
  `tau`/`beta`/`gamma`) — plus per-person estimator matrices, the
  `(K+1)^2` score correlation matrix, disattenuated subscore correlations, and
  added-value decisions (Haberman's `PRMSE_s > PRMSE_x`; Sinharay's 2010
  `+ 0.01` margin for augmentation, labeled — CRAN `CTTsub`'s relative rule is
  documented but not implemented). Formulas verified against the Appendix of
  Sinharay (2010, ETS RR-10-16) and the CRAN `subscore` R source read
  line-by-line; degenerate samples (alpha outside `(0, 1]`, zero variance,
  subscore collinear with the total) are rejected instead of propagating NaN.
  For LLM-as-a-Judge item-quality management this decides whether per-domain
  judge subscores add diagnostic value over the overall score. Rust-only
  numerics; the Python wrapper validates and marshals. Tests pin every
  reported statistic against literals from an independent NumPy transcription
  of the R semantics on an asymmetric fixture with mixed added-value
  outcomes, include rejection tests for the structural and degeneracy guards
  (the defensive computed-PRMSE-range guard is not separately exercised), a
  conditional dominance
  sweep on guard-passing random data, and a 500-rep `#[ignore]` Monte Carlo
  MSE comparison; three mutation spot-checks (dropped `m/(m-1)`, rowsum
  including the total column, `tau` numerator sign flip) were run and killed.
- **Kernel-smoothing nonparametric IRT** (`fast_mlsirm.ksirt_analysis`; new
  `mlsirm_core::ksirt`; Ramsay, 1991, as cited in Mazza et al., 2014).
  Estimates option characteristic curves by Nadaraya-Watson kernel regression
  (gaussian/quadratic/uniform kernels) of option indicators on rank-based
  ordinal ability estimates `qnorm(rank/(n+1))`, on an equally spaced
  evaluation grid, with Silverman-rule default bandwidths, plus expected item
  score and expected total score curves. Formulas verified against the
  KernSmoothIRT JSS paper (Mazza et al., 2014, Sections 2-2.3) and the
  KernSmoothIRT R/C++ package source read line-by-line; standard errors and
  cross-validation bandwidth selection are deliberately out of scope (the R
  implementation's SE accumulator is order-dependent and unverifiable from
  read sources). For LLM-as-a-Judge item-quality management this reveals
  non-monotone or poorly discriminating evaluation items without a parametric
  model. Rust-only numerics; the Python wrapper validates and marshals. Tests
  pin a hand-computed 4-person fixture (rank->theta qnorm literals, grid
  endpoints, Silverman constant), enforce structural invariants
  (row-sums-to-one with positive denominators, compact-support zeros,
  zero-denominator fallback), and include a 500-replication Monte Carlo
  recovery study (`#[ignore]`) under normal and skewed ability generation
  using the rank-invariance composition oracle.
- **Mokken scale analysis** (`fast_mlsirm.mokken_analysis`; new
  `mlsirm_core::mokken`; Mokken, 1971, as cited in van der Ark, 2007).
  Computes the Loevinger scalability coefficients `Hij`, `Hi`, `H` and their
  Mokken Z statistics, and partitions items into Mokken scales with the
  automated item selection procedure (AISP, "search normal"), with sample
  statistics and selection mechanics verified line-by-line against the mokken
  R package source (van der Ark, 2007; Straat et al., 2013): `Hij =
  S_ij/Smax_ij` with `Smax` from the comonotone (sorted-column) coupling,
  `Hi`/`H` as ratios of pairwise sums, and per-scale Bonferroni-adjusted Z
  gates. For LLM-as-a-Judge item-quality management this flags evaluation
  items that fail to scale (label 0) and detects multidimensional item pools
  before parametric calibration. Complete integer data required (dichotomous
  or polytomous). Rust-only numerics; the Python wrapper validates and
  marshals. Tests include a brute-force covariance oracle, an exact Guttman
  `H = 1` anchor, a hand-computed Z fixture, a Z-gate anchor whose deletion
  seeds a spurious scale (this test caught a real sign error in the normal
  quantile during development), a hand-constructed Criterion-1 design whose
  negative-`Hij` exclusion is the only active gate (mutation-verified), a
  two-cluster AISP recovery, and an `#[ignore]` 500-replicate Monte Carlo
  (normal + skewed traits).
- **Many-Facet Rasch Model (MFRM) rater-severity calibration** (`fast_mlsirm.fit_facets`;
  new `mlsirm_core::facets`; Linacre, 1989; Eckes, 2015). Fits
  `ln[P(k)/P(k-1)] = theta_p - d_i - c_j - f_k` — the rating scale model
  (Andrich, 1978) with a rater facet — to a `persons x items x raters` array with
  NaN-missing sparse judging plans. For LLM-as-a-Judge calibration this puts each
  judge's severity `c_j` on a common logit scale adjusted for item difficulty and
  respondent ability. Estimation is marginal-ML EM on a Gauss-Hermite grid
  (Bock & Aitkin, 1981), NOT Linacre's JMLE, and the docs say so: estimates match
  the Facets program only up to the JMLE-vs-MMLE difference. Identification:
  `theta ~ N(0,1)`, severities and thresholds centered to sum 0
  (`n_parameters = I + (J-1) + (K-2)`). Reports Linacre's connectedness
  diagnostic via union-find over the person-mediated item∪rater co-observation
  graph; `connected=False` means cross-component severity comparisons rest
  solely on the shared trait prior, not the rating design. Rust-only numerics;
  the Python wrapper validates and marshals. Tests include FD gradient anchors,
  the J=1 RSM-reduction identity, asymmetric-severity recovery, sparse and
  disconnected designs, and an `#[ignore]` 500-replicate Monte Carlo
  (normal + skew-normal traits) bounding severity bias and RMSE; a gradient
  sign-flip mutant was verified to fail 4 tests.
- **Warm's weighted-likelihood ability estimation for POLYTOMOUS items** (`fast_mlsirm.score_wle_poly`;
  new `score_wle_poly` in `mlsirm_core::scoring`; Warm, 1989). The library already had the full
  polytomous model family and polytomous EAP scoring, but its only bias-reduced ML ability estimator was
  dichotomous-only. EAP shrinks toward the population mean, which is exactly what individual score
  reporting must not do, so this closes a real gap rather than adding a third way to do the same thing.
  Solves `dlnL/dtheta + J/(2I) = 0` with `I = sum_k P'_k^2 / P_k` and `J = sum_k P'_k P''_k / P_k`
  accumulated over the person's observed items — the exact generalization of the shipped dichotomous
  `sum_i P' P''/(P Q)`, which is its two-category case. `J` is computed DIRECTLY, never as a derivative
  of `I`. GRM and GPCM; PCM is the GPCM path at `slope = 1`. RSM is deliberately NOT supported and the
  code says why: its fitted `(delta, shared tau)` parameterization is not convertible through any
  exposed API, since `rsm_logprobs` builds the equivalent intercepts internally and does not return
  them.
  **Verification status is stated in the code, because it bounds what may be claimed.** That the
  polytomous Warm correction is `J/(2I)` with `J = sum_k P'P''/P` is confirmed from the `catR` package's
  SOURCE, not from a primary paper — and `catR` keeps its Jeffreys-prior branch as a separate
  expression, so the two estimators are kept distinct here too (Magis & Raîche, 2012). Penfield and
  Bergeron (2005) treat the GPCM but their equations were not obtainable, so nothing here rests on them
  and they are not cited as a source. Separately, and PROVED in-repository rather than taken from a
  source: `J = I'` holds exactly for both shipped families. From `I' = 2J - T` one gets
  `J - I' = -E[l' l'']`, which vanishes because the GPCM's `l''` is category-free and the GRM's sum
  telescopes through `v_0 = v_K = 0`; checked numerically at 80-digit precision against fully numeric
  derivatives (relative `|J - I'| <= 1.1e-30`). The WLE therefore coincides with the Jeffreys modal
  estimate for these two families — but the identity is used ONLY as a test oracle and never as an
  implementation shortcut, because it fails for a graded model with per-boundary slopes and for the 3PL,
  both of which are pinned as negative controls.
  **Numerics.** Per-category quantities are formed division-free from the sigmoids — GPCM
  `P'_k/P_k = a(k - E)`, `P''_k/P_k = (P'_k/P_k)^2 - a^2 Var(k)`; GRM `P'_k/P_k = a(1 - s_k - s_{k+1})`,
  `P''_k/P_k = (P'_k/P_k)^2 - a^2(v_k + v_{k+1})` — so no category probability ever appears in a
  denominator and no probability floor is needed anywhere. The resulting GRM information is algebraically
  identical to the shipped `poly_item_information`, via `v_k - v_{k+1} = P_k(1 - s_k - s_{k+1})`.
  **Both polytomous log-likelihoods are log-concave, yet the weighted objective is genuinely
  multimodal**, because the Warm weight is not: a 3-item GPCM bank in the test suite has stationary
  points at `+0.0988`, `+0.3774` and `+1.3314` while `max lnL'' = -5.6e-5 < 0`, so a solver that brackets
  the first sign change from the left errs by 1.23 logits (2.36 on the GRM fixture). The global grid
  scan of `score_wle` is therefore reused unchanged, including its refusal to return an unresolved mode
  beyond 65,536 intervals; the grid demand additionally scales with `n_cat - 1`, documented as a derived
  worst-case margin for which no wrong-mode counterexample was reproduced.
  **Guards.** Eleven anchors, the important ones mutation-verified with the measured result recorded.
  `J == I'` is pinned with BOTH sides coming from different crate code paths (the accumulator versus a
  central difference of the shipped `poly_item_information`), and the reference magnitudes are pinned to
  1e-5 rather than merely asserted non-zero, so a zeroed or sign-flipped `jterm` fails. `K = 2`
  reproduces the dichotomous `score_wle` for both families — non-discriminating as a design argument,
  but the only anchor that catches a layout transpose, a `cat_params` stride bug or a missing
  chain-rule `a`. The two global-mode fixtures assert the returned value is the dominant mode and not
  the leftmost stationary point; a mutation that stops the scan at the first rise of `Phi` fails eight
  of the ten polytomous tests, and the narrower "take the leftmost stationary point" substitution fails
  the two global-mode fixtures specifically. The estimating equation is re-derived from finite
  differences of the log-probability routines alone, and the all-lowest/all-highest patterns are
  asserted finite alongside a check that the UNWEIGHTED score really does keep a constant sign there,
  so the "the MLE diverges" premise is verified rather than assumed.
  **One coverage limit is stated rather than papered over.** Because `J = I'` is EXACT for both shipped
  families, an implementation that replaced `J` with a numerical derivative of `I` would be
  behaviour-preserving and NO polytomous test can detect it — a mutation confirmed to leave the whole
  polytomous suite green. The discriminating anchors for that substitution live in the dichotomous
  suite, where a lower asymptote breaks the identity. The accompanying test therefore documents that the
  identity is family-specific (exhibiting a per-boundary-slope graded model and a 3PL where it fails,
  with measured relative gaps of 0.92/1.17 and 0.47) and is labelled a lemma about the ORACLE, not a
  test of the code.
  **Also corrects an error in the dichotomous WLE documentation shipped earlier in this release**: it
  claimed `J` coincides with `I'/2` for the 2PL/Rasch. The correct statement is `J = I'` exactly there,
  from `I' = 2J - T` with `T = sum_i P_i'^3 (1 - 2 P_i)/(P_i Q_i)^2`, and `T = J` only when
  `c = 0, d = 1` — which is why the weight is `sqrt(I)`. Fixed in the Rust, PyO3 and Python docstrings;
  the historical entry below is left as written. The identity is now PINNED by
  `wle_information_derivative_identity`, because the first attempt at this correction was itself wrong
  (it dropped the `(1 - 2 P)` factor from `T`, giving a value ~5x off at the 2PL) and nothing caught it.
  A formula asserted in prose and checked by no test is how that happens twice.

- **Uniform SIBTEST, the regression-corrected observed-score DIF procedure** (`fast_mlsirm.sibtest`;
  extends `mlsirm_core::dif`; Shealy & Stout, 1993). The third observed-score DIF procedure in the
  module and the only one that corrects the MATCHING CRITERION itself. Mantel-Haenszel and the logistic
  sweep both match on an observed number-correct score, which is unreliable: under IMPACT — a genuine
  group difference in ability — two examinees from different groups with the same OBSERVED score do not
  have the same expected TRUE score, because each regresses toward their own group's mean. Matching on
  the raw score therefore compares non-equivalent examinees. Item purification, added earlier in this
  release, cannot substitute for this: it changes WHICH items form the criterion, not the regression of
  true score on observed score, so a perfectly purified criterion is still biased. SIBTEST transports
  each group's conditional mean from that group's own Kelley-regressed true score
  `V*_Gk = [Xbar_G + alpha_G (k - Xbar_G)] / n_valid` to the unweighted midpoint of the two, using a
  per-level central difference taken over each group's OWN true-score scale at adjacent OBSERVED level
  positions, and compares the transported means under combined-sample weights renormalized over the
  retained strata. The valid and studied subtests are DISJOINT by construction — the opposite of the
  item-included Mantel-Haenszel default (Donoghue, Holland & Thayer, 1993), and a property of the
  estimator rather than an option. Per-group coefficient alpha is reported on every row because the
  correction divides by it.
  **The headline finding is unflattering and is documented as such.** Measured against
  `mantel_haenszel_dif` on identical simulated data, 500 replications per cell, no DIF planted so every
  rejection is a false positive: at zero impact MH holds .044 and SIBTEST .056; at impact 1.0 MH holds
  .046 while SIBTEST reaches **.086**. SIBTEST over-rejects in both cells and by roughly double under
  impact — the opposite of the ordering its motivation suggests. This is a property of the 1993
  estimator rather than a transcription slip (the closed-form anchors reproduce the TRANSCRIBED FORMULAS
  in exact rational arithmetic; `mirt` itself was never executed, so no cross-implementation agreement
  is claimed), whose standard error treats the ESTIMATED regression correction as fixed and so never
  charges the correction's own noise to the variance. It is precisely what Jiang and Stout's (1998)
  paper — "Improved Type I error control and reduced estimation bias for DIF detection using SIBTEST" —
  was written to fix, and that two-segment estimator is NOT implemented here. The docs accordingly
  recommend Mantel-Haenszel or the logistic sweep for routine screening and position SIBTEST as the way
  to obtain the regression-corrected *estimand*, reading `beta_uni` as an effect size rather than
  trusting `p_value` as a calibrated test. A 500-replication Monte-Carlo test pins that finding so the
  claim cannot rot.
  **Sign.** `beta_uni > 0` means harder for the FOCAL group — the OPPOSITE orientation to `mh_d_dif`
  and `std_p_dif` in the same module, kept rather than harmonised because published `|beta_uni|`
  cut-offs assume it, and asserted in both directions by a cross-module anchor whose product assertion
  would catch a future refactor that "harmonises" both conventions at once.
  **Provenance, stated because it bounds what the code may claim.** Every formula is transcribed from
  the reference implementation (Chalmers, 2012; the `SIBTEST` routine of `mirt`), which attributes them
  to Shealy and Stout (1993); the primary text was not consulted, so no comment cites the 1993 equations
  directly. One deliberate DIVERGENCE from that reference is marked in the code: where a neighbouring
  group-by-level cell is empty it imputes the mean to zero and feeds that fabricated value into the
  central difference, producing a finite but meaningless slope; this implementation drops the level.
  **Scope deliberately reduced after a spec-verification pass.** Crossing-SIBTEST is NOT built:
  Chalmers (2018) shows Li and Stout's (1996) hypothesis test is insufficient, no normal-theory referral
  for a crossing statistic is valid, and `logistic_dif`'s `S x G` interaction already covers crossing
  DIF against a standard 1-df null. No A/B/C letter class, because published cut-offs disagree and none
  was verified against its primary source — the same decision already taken for `delta_r2_uniform`. No
  purified variant, because purification needs a practical-significance predicate that no verified
  cut-off supports, and it would shorten the valid subtest and so lower the very reliability the
  correction divides by.
  **Guards.** Two CLOSED-FORM acceptance anchors, both derived in exact rational arithmetic and
  re-derived independently before the tests were written, assert to 1e-12: a single-stratum fixture
  (`beta = -1/10`, `sigma^2 = 23/1950`, `X^2 = 39/46`) and a five-level fixture pinning the weighting
  (`beta = -16993/88000`). The second is mandatory because the first is structurally blind to every
  weighting question — one retained stratum always carries weight 1 — and because the UNCORRECTED beta
  is `+0.11` under both weighting schemes, so the same assertion on an uncorrected statistic would prove
  nothing. Both were mutation-verified: substituting the observed mean for the true-score subtrahend
  yields `+0.26` (a sign flip), caught by both anchors; focal-group weights yield `-0.039932`, caught
  ONLY by the multi-level anchor. A third anchor pins a NON-CONTIGUOUS level vector to `1/128`, where
  both the hardcoded `2*alpha/n_valid` denominator and arithmetic `k +/- 1` indexing return `1/64` —
  every other fixture has contiguous levels, so without it both mutants survive the whole suite.
  Further anchors pin the strict `j_min` inequality on BOTH sides of its conjunction, all four corners
  of the empty-neighbour guard, the alpha gate on all four degenerate forms, per-group alpha against
  the direct KR-20 definition on a fixture whose groups differ in reliability (a pooled alpha survives
  any fixture where they do not), Benjamini-Hochberg in both directions plus `fdr_q` plumbing, and the
  disjointness of the criterion. That last one asserts the exact invariant rather than a proxy:
  complementing the studied item's column sends every conditional mean to `1 - Ybar` and every slope to
  `-M`, so `beta_uni` is exactly NEGATED and `se_beta` exactly invariant, while at least one other item
  must move — a criterion that ignored the data would be flip-invariant too.

- **Iterative item purification for the observed-score DIF procedures**
  (`fast_mlsirm.mantel_haenszel_dif_purified`, `logistic_dif_purified`; extends `mlsirm_core::dif`;
  Candell & Drasgow, 1988; Clauser et al., 1993; Holland & Thayer, 1988; Lord, 1980). Both DIF
  procedures added earlier in this release match examinees on the observed total score, which is
  itself built from the items under test — so when DIF items push a group's total in a CONSISTENT
  direction, the matching criterion is biased and clean items inherit spurious DIF (a false-positive
  inflation documented at both entry points). Purification breaks that circularity by re-running the
  sweep with the criterion rebuilt from the currently-unflagged ANCHOR set only: round 0 is the
  ordinary all-items sweep, each later round drops the flagged items from the criterion, and the loop
  stops when the flagged set comes back UNCHANGED from one round to the next (`converged = true`) or the
  round cap is hit. That is a stability test, not general cycle detection: a flagged set that oscillates
  between two states runs to the cap and is reported as `converged = false`, which is the honest answer
  rather than a spurious fixed point. The studied item is always added back into its own matching score
  even when it is not in the anchor, so every item is matched on `anchor UNION {studied}` — item-included
  matching is what makes the null-DIF condition hold (Holland & Thayer, 1988; Zwick, 1990), and it also
  makes round 0 identical to the unpurified sweep by construction: round 0 passes no anchor at all and so
  dispatches to the very same code path rather than to an all-true mask that merely evaluates the same.
  `PurifyConfig { max_rounds, min_anchor_items }` bounds the loop and refuses to purify below a usable
  anchor length (default 4), returning the last valid round with `converged = false` rather than a
  criterion built from a handful of items. Nothing is sized from the caller-supplied item count before
  that first sweep, since dimension validation lives inside the sweep and the count is untrusted at the
  FFI boundary.
  **Interpretation limits, documented at the API.** Purification REDUCES contamination, it does not
  remove it: the anchor is itself estimated, so the residual bias depends on how well round 0 separated
  the bank. More importantly the returned p-values carry **no Benjamini-Hochberg or Type-I guarantee** —
  the item set was selected using the same data, so the procedure is a screening device for flagging,
  not a calibrated test; the reported statistics must not be quoted as if they came from a single
  pre-registered sweep. Mantel-Haenszel purification also inherits MH's crossing-DIF blind spot and
  cannot repair it — an item MH never flags stays in the anchor every round and keeps contaminating the
  "purified" criterion. That blind spot is a property of the SIGNED AREA between the two curves over the
  matched ability distribution (Wang & Su, 2004) rather than of non-uniform DIF as such: a crossing at
  the centre of that distribution cancels and is invisible, while the same item with its crossing off
  centre leaves a net difference MH detects, so MH purification is unreliable rather than uniformly blind
  under non-uniform DIF. `logistic_dif_purified`, whose interaction term tests the crossing directly, is
  the variant to use there. **Guards.** A contamination fixture plants unidirectional DIF and asserts, in order, the
  PRECONDITION that the unpurified sweep really does false-flag clean items (so the test cannot pass on
  a fixture with nothing to fix), that purification strictly reduces those false flags, that the true
  positives are retained and are the items that left the anchor, and that the sweep statistics numerically
  changed; a clean bank asserts round 0 reproduces the shipped sweep EXACTLY (`n_anchor == n_items`,
  `rounds == 0`); and the cap and short-anchor exits are each pinned to report `converged = false`
  instead of silently returning a degenerate criterion. An adversarial implementation review then found
  those flag-counting fixtures could not see the arithmetic underneath them, and two structural anchors
  were added, each mutation-verified to fail on the defect it targets. (i) The returned `rows` must equal
  a fresh sweep against the returned `anchor`, swept over round caps, anchor floors and both matching
  conventions (which also covers `exclude_studied_item = true`, previously untested under purification):
  returning an earlier round's rows while reporting the final anchor is the highest-severity failure mode
  of a purification loop and is invisible to a "did it flag the right items" test, because intermediate
  rounds usually flag the same items. (ii) The purified row for an item must equal the ORDINARY sweep run
  on a reduced test consisting of exactly `anchor UNION {studied}` — an independent reference rather than
  the implementation's own arithmetic — checked for both a non-anchor item (the add-back branch) and an
  anchor item, with a deliberately NON-CONTIGUOUS anchor so an index-map error cannot hide behind a
  prefix. The anchor predicate is also pinned directly on all four ETS classes, since no simulated bank
  distinguishes "B or C" from "not A" (a clean 2PL never produces `Undefined`). The same review caught a
  key collision in the Python bindings: `logistic_dif_purified` wrote the loop's scalar convergence flag
  over `logistic_dif`'s PER-ITEM `converged` array, destroying it at the boundary — the loop flag is now
  `purify_converged` on both entry points — and found that both Python docstrings were written as
  `"""..."""` + a module constant, which is a `BinOp` rather than a constant expression, so the compiler
  never filled `__doc__` and the entire "not a calibrated test" caveat was invisible to `help()`.

- **Zumbo logistic-regression DIF, with non-uniform detection** (`fast_mlsirm.logistic_dif`; extends
  `mlsirm_core::dif`; Zumbo, 1999; Swaminathan & Rogers, 1990). Regresses each item response on the
  observed matching score `S`, the group `G`, and their interaction in three NESTED logistic models
  (`M0: b0 + b1 S`; `M1: + b2 G`; `M2: + b3 (S x G)`), fitted by IRLS/Newton. This closes the known blind
  spot of the Mantel-Haenszel procedure added earlier in this release: a stratified odds-ratio test can
  only see a *consistent* group advantage, so **crossing (non-uniform) DIF is invisible to it**, while
  the interaction term detects it directly. The 2-df `chi2_total = 2[ll(M2) - ll(M0)]` is the primary
  omnibus DIF decision (the value Benjamini-Hochberg adjusts); the 1-df components are descriptive
  follow-ups, and the module documents that `chi2_uniform = 2[ll(M1) - ll(M0)]` tests `b2` *assuming*
  `b3 = 0` — it is not the group term of the full model and is uninterpretable when non-uniform DIF is
  present, so the hierarchical entry order `S -> G -> S x G` is load-bearing. The effect size is the
  Nagelkerke (1991) pseudo-`R^2` change `delta_r2 = R2_N(M2) - R2_N(M0)` (with `ll_null` the
  intercept-only fit, and all four models fitted on one identical subsample so the normalizer is
  comparable), classified by Jodoin & Gierl (2001) — A `< 0.035`, B, C `>= 0.070` — and forced to A
  whenever the omnibus test is not BH-significant. The uniform-only `delta_r2_uniform` is reported
  without a letter class because those cut-offs were calibrated on the 2-df quantity; the more
  conservative Zumbo & Thomas (1997) cut-offs are documented as an alternative. **Robustness.** The
  matching score is mean-centered (the chi-squares are invariant, but the raw total leaves the `S x G`
  Gram near-singular); the design is rank-checked; the Newton step uses the *checked* solver and
  step-halving with a coefficient bound, so (quasi-)separation, a rank-deficient design, or
  non-convergence yield `NaN` statistics with `converged = false` and are never BH-flagged (a constant
  item is rejected outright). **Guards.** A SATURATED-DESIGN anchor (two-level score x binary group)
  pins the IRLS, log-likelihood, omnibus chi-square and Nagelkerke effect size against closed-form
  binomial arithmetic, plus the exact decomposition `chi2_uniform + chi2_nonuniform == chi2_total`; and
  the discriminating anchor plants a crossing item whose ICCs intersect at the common group ability
  mean, asserting the logistic test flags the interaction (with the uniform component non-significant)
  on the very item Mantel-Haenszel classifies as negligible, while a plain b-shift item shows the
  reverse pattern; the Jodoin-Gierl classifier is additionally pinned at its boundaries. Spec-verified
  (GO-WITH-MUST-FIXES applied), and an adversarial implementation review then fixed four further root
  causes: a NaN chi-square silently becoming `p = 1.0` in `chi2_sf` (which both misreported "no DIF" and
  let unfittable items dilute the Benjamini-Hochberg denominator), a convergence backstop that certified
  a bound-truncated separated fit as converged, a minimum-sample floor far too weak for the
  four-parameter model, and the unpinned classifier. Convergence uses the standard GLM relative-deviance
  test paired with a coefficient-bound separation check, since near the optimum the attainable score
  floor exceeds any usable absolute gradient tolerance. Same matching-criterion contamination as
  Mantel-Haenszel (see `logistic_dif_purified`), plus the logit-linearity-in-`S` assumption, both documented.

- **Rasch conditional maximum likelihood + Andersen's LR test** (`fast_mlsirm.fit_rasch_cml`,
  `andersen_lr_test`; new `mlsirm_core::rasch_cml`; Andersen, 1970, 1972, 1973). CML estimation of the
  dichotomous Rasch item difficulties: conditioning each response pattern on its raw score (the
  sufficient statistic for ability) ELIMINATES the person parameters, so the difficulties are estimated
  without any assumption on the ability distribution (Rasch's specific objectivity) and consistently at
  fixed test length — unlike the marginal-ML path (which must posit a `theta` distribution) or joint ML
  (inconsistent). The conditional log-likelihood
  `ln L_c = -sum_i s_i beta_i - sum_r n_r ln gamma_r(eps)` uses the elementary symmetric functions
  `gamma_r`; the ESF and its per-item/per-pair derivatives are computed by the numerically stable
  SUMMATION algorithm (a fresh forward pass `gamma_r += eps_j gamma_{r-1}` over the relevant item
  subset), avoiding the cancellation-prone subtractive difference recursion (Verhelst, Glas & van der
  Sluis, 1984). Newton on `beta` with sum-zero identification and a reduced-system solve; standard
  errors from the pseudoinverse of the conditional information; persons scoring `0` or `k` are dropped.
  Andersen's (1973) conditional likelihood-ratio test partitions the persons, fits CML within each group
  and pooled, and refers `2[sum_g llc_g - llc_pooled]` to `chi^2((G-1)(k-1))`. **Guards.** The
  summation-algorithm ESF (and its leave-one-out / leave-two-out passes) match brute-force subset sums;
  a deterministic finite-difference anchor pins the CML gradient AND Hessian (catching the
  `d eps/d beta = -eps` sign); the DEFINING person-distribution-free property is the primary anchor — the
  same `beta_hat` is recovered whether `theta` is `N(0,1)` or strongly right-skewed (a value-recovery
  test alone cannot separate CML from JML); and the Andersen LR does not over-reject Rasch data but
  rejects a planted group-specific difficulty shift, with the `df` and upper tail pinned. Spec-verified
  (GO-WITH-MUST-FIXES applied: the summation ESF over the difference recursion, dropping `r=0/k` persons,
  sum-zero centering, the reduced-Hessian SE, and reuse of `solve_small`/`chi2_sf`). An adversarial
  implementation review (faithfulness clean) then hardened two edge cases: `andersen_lr_test` surfaces a
  `converged` flag so a stalled fit's clamped `lr = 0` is not misread as a clean non-rejection, and the
  Python binding caps `n_groups` at 256 (u8 label range). Complete-data only; polytomous and missing-data
  CML are deferred. Exposed to Python as `fit_rasch_cml` and `andersen_lr_test`.

- **Warm's weighted likelihood estimation of ability** (`fast_mlsirm.score_wle`;
  `mlsirm_core::scoring::score_wle`; Warm, 1989). The bias-reduced maximum-likelihood ability estimator
  for unidimensional dichotomous items (2PL/3PL/4PL): it solves the weighted-likelihood estimating
  equation `dlnL/dtheta + J(theta)/(2 I(theta)) = 0` with the Warm correction
  `J = sum_i P_i' P_i''/(P_i Q_i)` computed DIRECTLY. Crucially `J` is *not* `I'(theta)/2` — the two
  coincide only for the 2PL/Rasch (`c=0, d=1`), where the weight is `sqrt(I)` (the Jeffreys prior); for
  the 3PL/4PL the second derivative carries `1-2s` while the information derivative carries `1-2P`, so a
  `sqrt(I)`-weighted estimator applies the wrong correction. Warm's estimator removes the leading
  `O(1/n)` MLE bias and — unlike the MLE, which is `+/-infinity` for the all-correct / all-incorrect
  pattern — yields a FINITE estimate for every response pattern. The estimate is the GLOBAL maximizer of
  the weighted log-likelihood (whose derivative is the estimating function), located by a grid scan plus
  a local root refinement — robust to the 3PL/4PL case where the weighted likelihood is multimodal
  (Samejima, 1973; Yen, Burket & Sykes, 1991) and a single bracketed root can select the wrong mode;
  it is clamped and flagged when the finite root falls beyond `theta_bound`, and a person with no
  observed items returns `NaN`. It reuses `item_information_4pl` for `I(theta)`; the SE is `1/sqrt(I)`.
  **Guards.** An estimating-equation root anchor is verified by INDEPENDENT finite-difference
  derivatives of `P` (so a `J` sign error in the analytic `P' P''` is not shared) across the 2PL, 3PL,
  and Rasch; a 2PL finiteness anchor confirms the perfect/zero patterns give finite, interior estimates
  with correct > incorrect; a monotonicity anchor confirms the estimate is nondecreasing in the
  number-correct score; a global-mode anchor confirms the multimodal-3PL worst case returns the dominant
  mode (`theta ~ -4.13`, ~10x more probable) rather than a minor root; an all-missing person returns
  `NaN`; and a `#[ignore]` >=500-rep Monte-Carlo confirms Warm's headline result — the WLE aggregate
  `|bias|` (~0.04) is an order of magnitude smaller than the boundary-clamped MLE's (~0.50), the gap
  widening at extreme abilities. Spec-verified (GO-WITH-MUST-FIXES: `J`-not-`I'`, the `I~0` division
  guard, plain natural-scale `a/b/c/d` rather than the log-alpha `ItemBank`); an adversarial
  implementation review then caught and fixed two defects the initial tests missed — the 3PL/4PL
  multimodality (a single bracketed bisection could return a non-dominant root; replaced by the
  global-mode grid search) and an all-missing person silently returning `theta = 0` (now `NaN`).
  Polytomous WLE (Penfield & Bergeron, 2005) is deferred. Exposed to Python as `score_wle` returning
  `theta`/`se`/`boundary`.

- **Mantel-Haenszel differential item functioning** (`fast_mlsirm.mantel_haenszel_dif`; new
  `mlsirm_core::dif`; Holland & Thayer, 1988). The observed-score, calibration-free DIF procedure — the
  complement to the parametric IRT-LR DIF (`dif_polytomous`): no item response model is fitted.
  Examinees are matched on the number-correct total (thin matching, studied item **included** by
  default per Donoghue, Holland & Thayer, 1993; `exclude_studied_item=True` uses the rest score), and
  per item the common odds ratio `alpha_MH = (sum_m A_m D_m / T_m)/(sum_m B_m C_m / T_m)` and the
  continuity-corrected MH chi-square `max(0, |sum A_m - sum E(A_m)| - 0.5)^2 / sum Var(A_m)` (with the
  hypergeometric `Var(A_m) = n_Rm n_Fm m1_m m0_m / (T_m^2(T_m-1))`, referred to `chi^2(1)`) are computed
  over the DIF-informative strata (all four `2 x 2` marginal totals positive). Reported on the **ETS
  delta metric** `MH_D-DIF = -2.35 ln(alpha_MH)` (negative = harder for the focal group) with the
  Robins-Breslow-Greenland (1986) standard error, the **ETS A/B/C** severity classification (Zieky,
  1993; A if not significant at .05 or `|D-DIF| < 1.0`, C if `|D-DIF| >= 1.5` and `|D-DIF| - 1.645 SE >
  1.0`, B otherwise — or `Undefined`/`"U"` when there are no informative strata or a degenerate odds
  ratio, *not* the affirmative "A"), and the **standardized P-DIF** companion (Dorans & Kulick, 1986)
  `sum_m n_Fm (P_Fm - P_Rm) / sum_m n_Fm` (focal minus reference, so its sign agrees with `MH_D-DIF`).
  Benjamini-Hochberg controls the across-item FDR; the p-value reuses `fitstats::chi2_sf`. **Guards.** A
  two-stratum hand-computed anchor pins `alpha_MH`, the continuity-corrected chi-square, `MH_D-DIF`, the
  RBG SE, `STD-P-DIF`, and the C label; a no-DIF symmetry anchor returns `alpha_MH = 1`, zero delta, and
  class A; a degenerate/perfect-separation case returns NaN statistics and `Undefined` (never A); and a
  planted uniform-DIF simulation flags the DIF item (class B/C, correct delta sign) while classifying
  the clean items A and agreeing with the parametric IRT-LR DIF on the flagged item. Because the MH
  chi-square is over-powered at large N and the studied item mildly contaminates the matching total, the
  A/B/C classification (not the raw significance) is the practical-significance guard — documented, with
  item purification (since shipped as `mantel_haenszel_dif_purified`) and SIBTEST (Shealy & Stout, 1993)
  noted as future work. Spec-verified
  (GO-WITH-MUST-FIXES: STD-P-DIF sign, `Var_m > 0` stratum gate, degenerate-odds guards, zero-clamped
  continuity numerator).

- **Dimension-agnostic IRT model API.** Item families are named by their
  response function rather than by UIRT/MIRT dimensionality:
  `fit_2pl`/`TwoPlFit`, `fit_grm`/`GrmFit`, and
  `fit_nominal`/`NominalResponseFit`. A single `model=` argument follows the
  R `mirt` convention (Chalmers, 2012): `model=1` denotes the unrestricted
  one-factor model, while `model=models.confirmatory(loading_pattern)` carries
  a confirmatory loading structure and derives its dimension count. The fitted
  result retains `n_dims` only as a derived read-only property of its model
  specification. Numeric exploratory requests above one factor fail explicitly;
  the Rust estimators do not yet implement unrestricted multidimensional loading
  rotation/identification, so a confirmatory anchor pattern is never relabeled
  as exploratory. The previous brand-new `*_mirt` entry points and module names
  were removed rather than retained as misleading aliases. See
  `python/fast_mlsirm/models.py` for the verified Chalmers (2012) APA reference
  and DOI.

- **Correlated latent factors for MH-RM** (`fit_mhrm(..., estimate_corr=True)`; Cai, 2010b confirmatory
  item factor analysis). Completes the MH-RM to a free latent CORRELATION matrix `Phi` (unit diagonal,
  `theta ~ MVN(0, Phi)`) rather than orthogonal factors. The Metropolis acceptance prior becomes
  `-0.5 (theta*^T Phi^{-1} theta* - theta^T Phi^{-1} theta)` (the symmetric proposal cancels; `Phi^{-1}`
  is recomputed by Cholesky each cycle), and the `D(D-1)/2` free off-diagonal correlations ascend the
  Gaussian-prior objective `Q(Phi) = -0.5[log|Phi| + tr(Phi^{-1} C)]` (`C` the imputed second moment,
  RAW/uncentered — `E[theta]=0` is fixed by identification) by a per-cycle Robbins-Monro GRADIENT step
  `offdiag += gain_k * sigma_grad(Phi, C)`, kept positive-definite by BACKTRACKING (halve the step
  until the rebuilt `Phi` is PD). This REUSES the `twopl.rs` correlation machinery verbatim
  (`build_corr`, `sigma_grad`, `chol_lower`, `sym_inv_logdet`, `flip_corr_dim`, now `pub(crate)`), the
  same helpers `fit_2pl`'s deterministic ECM correlation step uses — so the `Phi` estimation is shared,
  not duplicated. The per-dimension reflection flips the correlation off-diagonals for the flipped
  dimension (`corr(theta_d, theta_k) -> -corr`) together with the loading column and trait chain, so
  the reported `Phi` is consistent with the canonicalized signs. `estimate_corr=False` (default) keeps
  `Phi = I` and is BIT-IDENTICAL to the previous orthogonal fit (the acceptance prior branches to the
  original per-dimension `||theta*||^2 - ||theta||^2` on the same RNG stream). It is a gradient-RM (not
  Cai's Newton-preconditioned) covariance update — documented as such; it still converges almost surely
  to the same `Phi` root, only the (un-curvature-adapted) rate differs. **Guards.** A recovery test
  recovers an exchangeable `Phi` off-diagonal at a POSITIVE (`rho=0.4`), a near-PD-boundary
  (`D=3, rho=0.5`), and a NEGATIVE (`rho=-0.5`) correlation within Monte-Carlo tolerance, confirming the
  recovered matrix stays a valid PD correlation matrix; `estimate_corr=False` yields exactly the
  identity; and a `#[ignore]` 500-rep Monte-Carlo at the near-boundary `D=3, rho=0.5` reports the
  correlation RMSE/bias and would surface a persistent PD-backtracking stall. Exposed to Python as the
  `estimate_corr` argument and the `corr` field of `MhrmFit`.

- **Polytomous (GPCM) response family for MH-RM** (`fit_mhrm(..., family="gpcm", n_cat=K)`; Muraki,
  1992, generalized partial credit model estimated by the Cai, 2010 MH-RM). Extends the
  stochastic-approximation confirmatory item factor analysis from binary items to ordered polytomous
  items, scaling high-dimensional POLYTOMOUS IFA to a latent dimensionality where the deterministic
  `fit_gpcm` (Gauss-Hermite / QMC EM) is infeasible. Each item keeps a SINGLE multidimensional
  discrimination `a_i` (free on the confirmatory loading pattern) and gains `K-1` free UNORDERED step
  intercepts: `base_i = sum_{d in S_i} a_id theta_d` (NO intercept), `P(Y=k) = softmax_k(k*base_i +
  step_ik)` (`step_i0 = 0` pinned). The MH imputation likelihood is the inline log-softmax of the
  observed category (no per-node allocation), and the per-item RM step uses the **closed-form
  multinomial complete-data Hessian** `H = sum_p J_p^T (diag(P) - P P^T) J_p` (data-independent given
  `theta`, where the design row `J_p[k]` is `d psi_k / d param`: `k*theta_pd` for slope `a_id`, `[k==j]`
  for `step_j`) as BOTH the Robbins-Monro preconditioner AND the Louis positive term — NOT the BHHH
  score cross-product (which is the term Louis subtracts, so `H_BHHH - sum s s^T = 0` would give a
  degenerate SE). The complete-data score `sum_p J_p^T ([k==y_p] - P)` equals the deterministic
  `gpcm.rs`'s `[g_base*theta_d; g_intercepts]` with the integer scores fixed (`g_scores` dropped — what
  makes it GPCM, not nominal). The per-dimension reflection flips only the slope column and the trait
  chain — the UNORDERED steps are left INVARIANT (`base = k*sum a_d theta_d` is invariant under the
  joint `(a, theta)` sign flip), exactly as the deterministic `gpcm.rs`. `family="2pl"` (default) keeps
  the binary path **BIT-IDENTICAL** (the closed-form `log_sigmoid` score and `sum w X X^T` information
  are unchanged on the same RNG stream). GRM (Samejima cumulative-logit, ordered thresholds) is
  DEFERRED: its thresholds must stay strictly decreasing, which the deterministic `grm.rs` maintains by
  a backtracking line search a single stochastic RM Newton step cannot replicate (the standard path is a
  softplus threshold-gap reparametrization — future work). An adversarial implementation review found
  and fixed two defects the initial tests missed: the output/SE routing keyed on `n_free_cat == 1` as a
  "is 2PL" proxy, which mis-collapsed a legal `Gpcm { n_cat = 2 }` fit's single step into the 2PL
  `intercept`/`se_intercept` fields (now keyed on the model family); and the declared `MHRM_MAX_CAT`
  category cap was never enforced (an unbounded `n_cat` allocation vector — now validated). **Guards.** A
  deterministic finite-difference
  anchor pins the GPCM score AND the exact-multinomial information against the complete-data GPCM
  log-likelihood on an asymmetric cross-loader with a NEGATIVE loading and NON-MONOTONE steps (a sign
  flip, a transposed/dropped design slot, an over-collapsed step block, or BHHH-as-information all fail
  it), with an independent per-person score outer-product re-sum pinning the sign of the Louis
  missing-information subtraction; a `D=1` reduction test agrees with `poly::fit_poly_unidim(Gpcm)`
  (Bock-Aitkin quadrature) within Monte-Carlo tolerance; a `D=5` recovery (GH/QMC infeasible) recovers
  loadings, steps, and the negative cross-loader with correct sign; a reflection-FIRES test witnesses
  the canonicalization flipping a negative anchor while leaving the steps un-swept; the validation
  rejects out-of-range responses and any never-observed category (an unidentified step); and a
  `#[ignore]` 500-rep Monte-Carlo (normal + right-skew traits, `D=2` and `D=5`, `K=3`) reports the
  loading/step RMSE and bias. Exposed to Python as the `family`/`n_cat` arguments and the
  `step`/`se_step`/`n_cat` fields of `MhrmFit`.

- **High-dimensional confirmatory 2PL by Metropolis-Hastings Robbins-Monro** (Cai, 2010).
  `fit_mhrm(responses, model=...)` fits the general compensatory multidimensional 2PL
  (`P(X_ij = 1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`, `theta ~ MVN(0, I_D)`) — the
  same model as `fit_2pl` — by a STOCHASTIC-approximation EM that scales to a latent dimensionality
  where the deterministic `q^D` Gauss-Hermite grid and the QMC E-step of `fit_2pl` are infeasible
  (`n_dims` up to 64). Each cycle (1) IMPUTES each person's `theta` by a short PERSISTENT
  (warm-started) symmetric random-walk Metropolis chain from its current posterior
  `pi_j(theta) prop phi(theta; 0, I) prod_i P_i(y_ij | theta)` — the acceptance ratio is the pure
  Metropolis posterior ratio (the symmetric proposal cancels), the proposal SD is auto-tuned toward a
  target acceptance during burn-in, and the chain carries across cycles so no per-cycle burn-in is
  needed; and (2) takes one Robbins-Monro stochastic-Newton step
  `xi <- xi + gain_k Gamma_k^{-1} s_k` on the complete-data score `s_k` (Fisher's identity gives an
  unbiased-in-the-limit Monte-Carlo estimate of the marginal score) and the RM-smoothed information
  `Gamma_k = Gamma_{k-1} + gain_k (H_k - Gamma_{k-1})`. Because the item blocks are conditionally
  independent given `theta`, the score, information, and RM step are BLOCK-DIAGONAL by item, and the
  per-item work is the CLOSED-FORM logistic gradient `X'(y - P)` and information `X'WX` — no
  quadrature, `D`-independent per-node cost (reusing `mmle::{log_sigmoid, sigmoid_stable}` and
  `poly::solve_small`). The gain follows a constant-gain burn-in (a Metropolis-Hastings stochastic EM
  that random-walks into the MLE neighbourhood) then a decreasing `gain_k = 1/(k - k0)^alpha`
  (`sum gain = inf`, `sum gain^2 < inf`, Robbins & Monro 1951) that converges almost surely to a
  marginal-score root. Convergence is WINDOWED (the running mean of `||xi^(k) - xi^(k-1)||` over the
  last `w` cycles falls below `tol`) — MH-RM iterates are non-monotone by design, so no
  likelihood-decrease guard is used. **Identification.** Unit trait variances fix the loading scale,
  `E[theta] = 0` the intercepts, and a PURE single-dimension anchor item per dimension pins the
  rotation; the per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` is likelihood-
  invariant, and because the stochastic iterates could otherwise drift between the two mirror modes
  and corrupt the RM RUNNING AVERAGE of the loadings, the canonical sign (largest pure anchor
  positive) is enforced IN-LOOP every cycle — flipping the loading column, the persistent `theta`
  chain, and the averaged trait together — and once more at the end (mutation-verified: disabling the
  flip makes the reflection-fires test fail on all three sign checks). Loadings are UNCONSTRAINED so
  reverse-keyed / negative cross-loadings are representable. **Standard errors.** The Louis (1982)
  identity `I_obs = E[-d^2 l_c] - Var[d l_c]` gives per-item observed-information SEs, accumulated by
  a parallel RM filter (`sum_p (w_p - r_p^2) X_p X_p'`) over the convergence stage; where a
  finite-sample Louis block is not positive-definite the block falls back to the complete-data
  (Fisher) information (a conservative SE). **Guards.** A deterministic finite-difference anchor pins
  the per-item score and information against numerical derivatives of the complete-data logistic
  log-likelihood on an ASYMMETRIC D=2 cross-loader with a negative loading (catching sign, layout, and
  dims-map bugs a centered value-recovery test would not); the D=1 fit agrees with the established
  deterministic unidimensional MMLE (`mmle::fit_mmle_2pl`) within Monte-Carlo tolerance; a **D=6**
  recovery (3 pure anchors per dimension + a negative cross-loader, GH/QMC infeasible) recovers the
  loadings and per-dimension traits; the reflection-fires test drives a weak reverse-keyed pure anchor
  against a strong positive cross-loader so raw MH-RM lands the anchor negative and canonicalization
  must fire; and `validate` rejects rotationally-degenerate patterns, non-binary responses, and
  `burn_in >= max_cycles`. This first release fits the ORTHOGONAL 2PL (`Sigma = I`); a free latent
  correlation matrix and the polytomous item families are natural extensions of the same loop. Compute
  lives in `mlsirm_core::mhrm::fit_mhrm`; exposed to Python as `fit_mhrm` / `MhrmFit` via the
  `model=` specification API.

- **Confirmatory MULTIDIMENSIONAL generalized partial credit model** (Muraki, 1992).
  `fit_gpcm(responses, n_cat, model=...)` fits ORDERED polytomous categories with a SINGLE
  multidimensional discrimination vector per item and INTEGER category scores, completing the
  polytomous-MIRT trio (`fit_nominal` / `fit_grm` / `fit_gpcm`). Item `i` has a free slope `a_i` (free
  on the confirmatory 0/1 loading pattern from `model=models.confirmatory(...)`, items x D) and
  `n_cat-1` category step intercepts `gamma_i`, with `psi_k = k * (sum_{d in S_i} a_id theta_d) +
  gamma_i,k`, `gamma_i,0 = 0` pinned, and `P(Y_i = k | theta) = softmax_k(psi_k)`, `theta ~ MVN(0,
  I_D)`. This is the `a_ikd = k a_id` INTEGER-scoring restriction of the multidimensional nominal
  model in a distinct single-slope parametrization — NOT a mode of `fit_nominal` (which optimizes free
  per-category slopes), so it warrants its own estimator; and it is the ADJACENT-category-logit
  counterpart of the cumulative `fit_grm`. Unlike the GRM's thresholds, the GPCM steps are UNORDERED
  (the softmax is finite for any real `gamma`, so no ordering constraint exists or is imposed). It
  reduces to the unidimensional GPCM (`poly::fit_poly_unidim(PolyModel::Gpcm)`) at `D = 1` (within
  optimizer tolerance and up to reflection — NOT bit-exact, because `fit_poly_unidim` forces `a > 0`
  via a `log a` parametrization while the confirmatory model uses an UNCONSTRAINED slope so
  reverse-keyed / negative cross-loadings are representable). Estimated by Bock-Aitkin marginal MLE
  over the D-dim latent grid, REUSING the compensatory-MIRT node machinery (`nodes::build_xi_nodes`):
  `node_rule = "gh"` uses the `q^D` Gauss-Hermite grid (`D <= 3`), `"qmc"`/`"mc"` use `xi_points`
  Halton / Monte-Carlo draws (`D <= 6`, Jank 2005 QMC-EM), and the GPCM softmax cell of
  `poly::gpcm_logprobs` / `gpcm_node_gradient`. The per-item M-step is a finite-difference-Hessian
  Newton over `[a_{d0}..a_{d,L-1}, gamma_1..gamma_{M-1}]`, byte-for-byte the ascent of
  `poly::m_step_item` (ridge = Hessian conditioning only, not a prior), with the GPCM node gradient
  chained to the multidimensional slope (`d/da_id = sum_node g_base theta_d`, `d/dgamma_j = sum_node
  g_intercepts[j]`). Category scores are FIXED integers `0..n_cat-1` (that fixity is what makes the
  model GPCM rather than nominal), so the free per-category slope gradient returned by the shared cell
  (`g_scores`) is DROPPED — only the single `base` slope and the step intercepts are estimated. Init is
  `gamma_k = ln(freq_k / freq_0)` (a plain marginal log-odds, NOT a cumulative GRM-style boundary). EM
  uses the SIGNED monotonic-decrease stopping guard (a likelihood decrease errors, not the
  compensatory MIRT's `.abs()` check). **Identification.** Unit trait variances + a PURE
  single-dimension anchor item per dimension pin the rotation to the coordinate axes; the per-dimension
  reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` leaves `base` — hence every step and category
  probability — INVARIANT, so it is CANONICALIZED (as for the GRM / compensatory MIRT, and unlike the
  nominal, whose per-category slopes make the anchor sign ambiguous): dimension `d` is flipped so its
  largest-magnitude pure anchor loads positively, negating that dimension's slope column AND the trait
  `theta_d` but NOT the steps. `validate` rejects a rotationally-degenerate pattern (no pure anchor),
  an out-of-range category, and ANY unobserved category for an item, with a `nodes x items x n_cat`
  count-table cap and the rule-dependent D / q / xi_points bounds. **Guards.** The D=1 anchor recovers
  `fit_poly_unidim(Gpcm)`'s slope and steps within tolerance; a deterministic finite-difference anchor
  pins every per-(dimension, step) gradient slot on a fixed node set at D=2 (GH) AND D=4 (Halton) with
  a NON-IDENTITY dims map, M>=4 categories, deliberately NON-MONOTONE step values (unordered steps have
  no ordering canary, so the anchor exercises the free-step estimator directly) and distinct random
  per-category counts; because that FD anchor is map-invariant, a SEPARATE deterministic
  objective-value assertion at D=4 (dims `[0,2,3]`) pins the node-column dims map by computing
  `base = sum_t a_t node[dim_t]` and the GPCM log-probabilities BY HAND with LITERAL integer scores and
  matching the estimator's internal value to `< 1e-9` (the QMC path is never exercised by the D<=3
  recovery / MC); a reflection-FIRES test is constructed so the RAW EM mode lands the pure anchor
  NEGATIVE (a WEAK reverse-keyed pure anchor plus a STRONG positively-keyed cross-loader that dominates
  the dim0 orientation), so canonicalization MUST fire — asserting the anchor ends positive, the
  co-loader ends negative, the trait axis is sign-flipped (theta correlates negatively with the truth
  on the reflected dimension), and the steps are unchanged; mutation-verified (disabling the flip fails
  all three sign checks). A D=2 recovery carries a genuinely NEGATIVE cross-loader on a
  positively-anchored dimension (asserted `< -margin`) and recovers the unordered steps by RMSE. A
  Monte-Carlo (`D in {2, 3}`, pure anchors + sign-varied cross-loaders, `n_cat = 4`, GH `q = 15/11`,
  `N = 2500/2000`) recovers the loadings near-unbiased under a normal trait (loading RMSE ~0.08-0.09,
  bias ~0.00-0.01; step RMSE ~0.06-0.07) with the expected mild attenuation under a
  per-dimension-standardized right-skew trait (loading RMSE ~0.10-0.11, bias ~-0.04; step RMSE ~0.14),
  per-dimension trait EAP correlation ~0.74-0.77 and 100% convergence, EM monotone every replication
  (40-replication pilot; the committed `#[ignore]` test runs 500). Compute lives in
  `mlsirm_core::gpcm::fit_gpcm`; exposed to Python as `fit_gpcm` / `GpcmFit`.

- **Confirmatory MULTIDIMENSIONAL graded response model** (Samejima, 1969; Muraki & Carlson, 1995).
  `fit_grm(responses, n_cat, model=...)` fits ORDERED polytomous categories with a SINGLE
  multidimensional discrimination vector per item and ordered category boundaries: item `i` has a
  free slope `a_i` (free on the confirmatory 0/1 `loading_pattern`, items x D) and `n_cat-1` ORDERED
  boundary intercepts `beta_i`, with `P(Y_i >= k | theta) = sigmoid(sum_{d in S_i} a_id theta_d +
  beta_i,{k-1})`, `theta ~ MVN(0, I_D)`. This is the ORDERED counterpart of the multidimensional
  nominal model and the polytomous generalization of the compensatory MIRT; it reduces to the
  unidimensional GRM (`poly::fit_poly_unidim(PolyModel::Grm)`) at `D = 1` (within optimizer tolerance
  and up to reflection — NOT bit-exact, because `fit_poly_unidim` forces `a > 0` via a `log a`
  parametrization while the confirmatory model uses an UNCONSTRAINED slope so reverse-keyed / negative
  cross-loadings are representable). Estimated by Bock-Aitkin marginal MLE over the D-dim latent grid,
  REUSING the compensatory-MIRT node machinery (`nodes::build_xi_nodes`): `node_rule = "gh"` uses the
  `q^D` Gauss-Hermite grid (`D <= 3`), `"qmc"`/`"mc"` use `xi_points` Halton / Monte-Carlo draws
  (`D <= 6`, Jank 2005 QMC-EM), and the GRM cumulative-logit cell of `poly::grm_logprobs` /
  `grm_node_gradient`. The per-item M-step is a finite-difference-Hessian Newton over
  `[a_{d0}..a_{d,L-1}, beta_1..beta_{M-1}]`, byte-for-byte the ascent of `poly::m_step_item` (ridge =
  Hessian conditioning only, not a prior), with the GRM node gradient chained to the multidimensional
  slope (`d/da_id = sum_node g_base theta_d`, `d/dbeta_j = sum_node g_thr[j]`). The ORDERED-threshold
  constraint is maintained WITHOUT an explicit reparametrization: every adjacent boundary pair is a
  middle category whose log-probability goes non-finite the instant the pair inverts (`0*NaN=NaN` so a
  zero expected count cannot mask it), so the backtracking line search — which rejects any non-finite
  step — keeps `beta` fully ordered by adjacency + transitivity. EM uses the SIGNED
  monotonic-decrease stopping guard (a likelihood decrease errors, not the compensatory MIRT's
  `.abs()` check). **Identification.** Unit trait variances + ordered thresholds + a PURE
  single-dimension anchor item per dimension pin the rotation to the coordinate axes; the
  per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` leaves `base` — hence every
  threshold and category probability — INVARIANT, so it is CANONICALIZED (unlike the nominal, whose
  per-category slopes make the anchor sign ambiguous): dimension `d` is flipped so its
  largest-magnitude pure anchor loads positively, negating that dimension's slopes AND the trait
  `theta_d` but NOT the thresholds. `validate` rejects a rotationally-degenerate pattern (no pure
  anchor), an out-of-range category, and ANY unobserved category for an item (a GRM boundary would
  diverge), with a `nodes x items x n_cat` count-table cap and the rule-dependent D / q / xi_points
  bounds. **Guards.** The D=1 anchor recovers `fit_poly_unidim(Grm)`'s slope and thresholds within
  tolerance (all-positive DGP, the domain where its `log a` is correctly specified); a deterministic
  finite-difference anchor pins every per-(dimension, threshold) gradient slot on a fixed node set at
  D=2 (GH) AND D=4 (Halton) with a NON-IDENTITY dims map, M>=4 categories, STRICTLY-DECREASING
  thresholds (gaps >> the FD step, since the GRM cell NaNs on an inverted boundary) and distinct
  random per-category counts; because that FD anchor is map-invariant, a SEPARATE deterministic
  objective-value assertion at D=4 (dims `[0,2,3]`) pins the node-column dims map by computing
  `base = sum_t a_t node[dim_t]` and the GRM log-probabilities BY HAND and matching the estimator's
  internal value to `< 1e-9` (the QMC path is never exercised by the D<=3 recovery / MC); a
  reflection-FIRES test drives a reverse-keyed largest pure anchor and asserts it ends positive, a
  co-loader ends negative, and the thresholds are unchanged and still ordered; a D=2 recovery carries
  a genuinely NEGATIVE cross-loader on a positively-anchored dimension (asserted `< -margin`) with
  strictly-ordered recovered thresholds. A Monte-Carlo (`D in {2, 3}`, pure anchors + sign-varied
  cross-loaders, `n_cat = 3`, GH `q = 15/11`, `N = 2500/2000`) recovers the loadings near-unbiased
  under a normal trait (loading RMSE ~0.10, bias ~0.00-0.01; threshold RMSE ~0.05-0.06) with the
  expected mild attenuation under a per-dimension-standardized right-skew trait (RMSE ~0.17/0.18,
  bias ~-0.12/-0.13), per-dimension trait EAP correlation ~0.63-0.70 and 100% convergence, EM
  monotone and thresholds ordered every replication (40-replication pilot; the committed `#[ignore]`
  test runs 500). Compute lives in `mlsirm_core::grm::fit_grm`; exposed to Python as
  `fit_grm` / `GrmFit`.
- **Confirmatory MULTIDIMENSIONAL nominal response model** (Bock, 1972; Thissen, Cai, & Bock,
  2010). `fit_nominal(responses, n_cat, model=...)` fits unordered polytomous categories
  with CATEGORY-SPECIFIC multidimensional discrimination: category `k` of item `i` has a free slope
  vector `a_ik` (free on the confirmatory 0/1 `loading_pattern`, items x D) and intercept `c_ik`,
  and `P(Y_i = k | theta) = softmax_k(sum_{d in S_i} a_ikd theta_d + c_ik)` with the baseline
  category `0` pinned `a_i0 = 0, c_i0 = 0`, `theta ~ MVN(0, I_D)`. This generalizes the
  unidimensional `poly::fit_nominal` to D latent dimensions, and reduces to it EXACTLY at `D = 1`
  (the same general free-`a_k` parametrization). Estimated by Bock-Aitkin marginal MLE (EM) over the
  D-dimensional latent grid, REUSING the compensatory-MIRT integration machinery: `node_rule = "gh"`
  uses the `q^D` Gauss-Hermite product grid (`D <= 3`); `"qmc"`/`"mc"` use `xi_points` Halton /
  Monte-Carlo draws (`D <= 6`), the quasi-Monte-Carlo EM of Jank (2005). The per-item M-step is a
  Newton on the concave multinomial-logit complete-data objective, byte-for-byte the
  finite-difference-Hessian ascent of `poly::nominal_m_step` (the ridge is Hessian conditioning only,
  NOT a parameter prior, so the fit is genuine MML and the D=1 reduction is bit-exact), generalized
  so the softmax residual `resid_k = r_k - n P_k` drives `d/dc_ik = sum_node resid_k` and
  `d/da_ikd = sum_node resid_k theta_d`. EM uses `fit_nominal`'s relative-tolerance stopping with a
  SIGNED monotonic-decrease guard (a likelihood decrease errors, rather than the compensatory MIRT's
  `.abs()` check which would accept one as convergence). **Identification.** Baseline category +
  unit trait variances + a PURE single-dimension anchor item per dimension pin the rotation to the
  coordinate axes: a pure anchor forces every one of its category slopes onto the axis, so an
  orthogonal trait rotation must send that axis to `+-e_d`, and the confirmatory labels forbid axis
  permutation — leaving only a per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)`,
  which (as in `fit_nominal`) is NOT canonicalized; recovery is assessed up to it. `validate` rejects
  a rotationally-degenerate pattern (no pure anchor), an out-of-range category, and — a guard
  `fit_nominal` lacks — ANY unobserved category for an item (its intercept would diverge and its D
  slopes be unidentified), plus a `nodes x items x n_cat` count-table cap and the rule-dependent
  D / q / xi_points bounds. **Guards.** The D=1 anchor reproduces `fit_nominal`'s scores/intercepts
  and whole loglik trace bit-exactly (< 1e-9); a deterministic finite-difference anchor pins EVERY
  per-(category, dimension) gradient component on a fixed node set at D=2 (GH) AND D=4 (Halton) with
  a NON-IDENTITY dims map and distinct random per-category counts (catching a category<->dimension
  transposition the D=1 reduction cannot see); a D=2 recovery carries a genuinely NEGATIVE
  cross-loader slope AND two OPPOSITE-sign sibling categories on the same dimension (catching a
  collapse of the free per-category slopes to a shared scalar discrimination); and baseline /
  off-pattern entries are asserted EXACTLY `0.0` with a free-parameter-count invariant. A
  Monte-Carlo (`D in {2, 3}`, pure anchors + sign-varied cross-loaders, `n_cat = 3`, GH
  `q = 15/11`, `N = 2500/2000`, assessed up to per-dimension reflection) recovers the category
  slopes near-unbiased under a normal trait (slope RMSE ~0.12 at `D = 2` / ~0.13 at `D = 3`, bias
  ~0.00-0.01) with the expected mild attenuation under a per-dimension-standardized right-skew trait
  (RMSE ~0.21/0.22, bias ~-0.09), per-dimension trait EAP correlation ~0.61-0.67 and 100%
  convergence, EM monotone every replication (the figures are a 40-replication pilot; the committed
  `#[ignore]` test runs 500). Compute lives in
  `mlsirm_core::nominal::fit_nominal`; exposed to Python as `fit_nominal` /
  `NominalResponseFit`.

- **Confirmatory compensatory multidimensional 2PL (MIRT), orthogonal or correlated**
  (Reckase, 2009; Bock, Gibbons, & Muraki, 1988).
  `fit_2pl(responses, model=...)` fits
  a general COMPENSATORY multidimensional 2PL in which an item may load FREELY on several
  latent dimensions, which trade off ADDITIVELY inside a single logit:
  `P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`, `theta_j ~ MVN(0, I_D)`,
  where `S_i` is item `i`'s loading set from a 0/1 confirmatory pattern (items x dimensions).
  This is Reckase's compensatory M2PL / the full-information item factor model, distinct from
  the existing simple-structure `Mirt` (one dimension per item) and the orthogonal bifactor
  (one primary + one general per item): arbitrary within-item cross-loadings break the
  simple-structure quadrature factorization, so it is a dedicated estimator (standalone
  `mlsirm_core::twopl`) with the full `q^D` product Gauss-Hermite grid (`D <= 3`). Estimated by
  marginal-ML EM: the E-step is streamed per person (no `N x q^D` posterior materialized), and
  each item M-step is an `(n_i + 1)`-dimensional Newton generalizing `fit_mmle_2pl`'s 2x2 — the
  ridged, positive-definite `-Hessian` block solved by Gaussian elimination with a backtracking
  line search that keeps the marginal loglik monotone. Loadings are **not** constrained
  non-negative (reverse-keyed and suppressor cross-loadings are representable); the
  per-dimension sign is fixed by a reflection anchor. **Latent traits:** `theta ~ MVN(0,
  Sigma)` — orthogonal (`Sigma = I`) by default, or with `estimate_corr = true` the
  inter-factor **correlation matrix is estimated**: the standard grid is mapped through
  `chol(Sigma)` (`theta_g = L z_g`, a measure-preserving change of variables that reuses the
  product-GH weights and the item M-step verbatim), and the `D(D-1)/2` free correlations ascend
  the Gaussian-prior objective `-0.5[log|Sigma| + tr(Sigma^{-1} C)]` (`C` the posterior second
  moment, accumulated via the per-node marginal mass so it adds nothing to the E-step order)
  with backtracking + a full-matrix positive-definite guard, keeping EM monotone; the reflection
  anchor also negates the flipped dimension's correlation off-diagonals. A deterministic
  finite-difference anchor pins the correlation gradient (`D=2` and `D=3`); a known-`Sigma`
  (`rho=0.5`) recovery with a reflection-triggering negative anchor confirms the sign flip; and
  a 500-rep MC recovers the correlations essentially UNBIASED against the realized sample
  correlation (correlation RMSE ~0.035-0.05, bias ~0.0005 under the normal model / ~0.017 under
  the NORTA right-skew arm), 100% convergence with every fitted `Sigma` strictly interior.
  `D > 3` (coarser GH or QMC) remains deferred. Identification is enforced by
  `validate`: every dimension must have a PURE single-loading anchor item, so
  rotationally-degenerate patterns (e.g. all-ones) are rejected rather than returning a point
  on a non-identified ridge. Verified with the N(0,I) grid-moment identities, a DETERMINISTIC
  finite-difference anchor pinning the full item gradient AND the off-diagonal cross-Hessian
  (the local->pattern-dimension map) to `< 1e-4`, an exact reduction to `fit_mmle_2pl` at `D=1`
  (`gh_rule(41)` is the same grid; loadings/intercepts agree to `< 1e-2`), and a non-trivial
  `D=2` recovery with asymmetric loadings INCLUDING genuinely negative ones (recovered with
  correct sign). A 500-replication Monte-Carlo (`D in {2,3}`, `N = 3000/2000`, confirmatory
  pattern with pure anchors + cross-loaders) recovers the loadings essentially UNBIASED under
  the correctly-specified normal trait (loading RMSE ~0.10 at `D=2` / ~0.12 at `D=3`, bias
  ~0.006) and shows the expected mild loading attenuation under a per-dimension-standardized
  right-skew trait (shape misspecification; RMSE ~0.12/0.16, bias ~-0.06/-0.10), with
  per-dimension trait EAP correlation ~0.67-0.72 and 100% convergence, EM monotone every
  replication. Exposed to Python as `fit_2pl` / `TwoPlFit`.
- **`D > 3` confirmatory compensatory MIRT via quasi-Monte-Carlo EM** (Jank, 2005). The
  compensatory MIRT above was capped at `D <= 3` by its `q^D` Gauss-Hermite product grid;
  `fit_2pl` now takes a `node_rule` (`"gh"` default, or `"qmc"`/`"mc"`) that swaps
  the E-step integration nodes for a **Halton quasi-Monte-Carlo** (or seeded Monte-Carlo) rule,
  reaching `D = 4, 5, 6` (the Halton prime axes). This is Jank's (2005) QMC-EM: the E-step integral
  `int p(x|theta) phi(theta) dtheta` is evaluated at `xi_points` points drawn from the prior
  (Halton radical inverse mapped through the inverse-normal CDF, equal weights `1/xi_points`)
  instead of the product grid, and the node set is built ONCE before the EM loop, so the per-item
  `(n_i+1)`-dim Newton M-step and the correlated-`Sigma` ECM step are byte-for-byte the same code on
  the swapped nodes. The reused node generator (`mlsirm_core::nodes::build_xi_nodes`, shared with the
  marginal QMC-EM family) is parity-tested; its Gauss-Hermite arm is bit-identical to the existing
  product grid, so the `"gh"` path is unchanged bit-for-bit and every prior MIRT test passes verbatim.
  Both the orthogonal and the correlated-`Sigma` (Cholesky node-map `theta_g = L z_g`) paths carry
  over to `D > 3`. **Monotonicity.** With `Sigma = I` the nodes never move, so the orthogonal fit is
  monotone in the QMC-approximated marginal likelihood; the correlated `Sigma` M-step reparametrizes
  the node cloud, so that fit is monotone only up to the QMC quadrature error (overall ascent with
  per-step wobble ~1e-5 relative that shrinks as `xi_points` grows) — use the orthogonal path or a
  larger `xi_points` when strict monotonicity matters. Validation is rule-dependent: `"gh"` keeps
  `D <= 3` and the `q^D <= 200_000` node cap; `"qmc"`/`"mc"` cap `D <= 6` (the Monte-Carlo node
  builder has no internal cap, so this bound is its sole guard) and bound `xi_points`
  (`1..=200_000`, with checked `xi_points * n_items` and `xi_points * n_dims` allocations); `q`
  applies only to `"gh"`, `xi_points`/`xi_seed` only to `"qmc"`/`"mc"`. **Guards.** Beyond the
  reused-grid regression, a deterministic layout pin asserts `build_xi_nodes(Halton).grid[j*D+k] ==
  inv_normal_cdf(radical_inverse(j+1, prime_k))` at `D = 4` (independently fixing the prime-to-axis
  assignment, the index skip, and the row-major layout that a value-recovery test cannot see); the
  QMC weights are pinned to `-ln(n)` (invisible to every fit-level test since they cancel in the
  self-normalized posterior); a deterministic finite-difference anchor pins the analytic gradient and
  full cross-Hessian on a FIXED Halton grid at `D = 4` with a non-identity `dims` map; the reduction
  anchor is TWO-SIDED (a `D = 2` QMC fit agrees with the GH fit within QMC error AND differs
  bit-wise, so a silent GH fallback is caught); and the reflection anchor is exercised with a
  reverse-keyed largest anchor. **Accuracy.** A Monte-Carlo (`D in {4, 5}`, confirmatory pattern with
  pure anchors + alternating-sign cross-loaders, Halton `xi_points = 4000/6000`, `N = 2000/1500`)
  under a correctly-specified normal trait recovers the loadings near-unbiased (loading RMSE ~0.13 at
  `D = 4` / ~0.17 at `D = 5`, bias ~0.01) and shows the expected mild attenuation under a
  per-dimension-standardized right-skew trait (shape misspecification; RMSE ~0.16/0.21, bias
  ~-0.07/-0.09), with per-dimension trait EAP correlation ~0.58-0.64 and 100% convergence, EM
  monotone every replication (the reported figures are a 50-replication pilot; the committed
  `#[ignore]` test runs 500). QMC carries an `O(N^{-1} (log N)^D)` finite-node bias that grows with
  `D` (the higher-prime Halton axes degrade), so `D = 5, 6` and the correlated `Sigma` off-diagonals
  need materially larger `xi_points`; `xi_seed` (nonzero by default) applies a Cranley-Patterson
  shift that partly de-correlates the higher axes. Exposed to Python as `fit_2pl(...,
  node_rule=, xi_points=, xi_seed=)`.
- **Shared-Q sequential G-DINA for polytomous responses** (Ma & de la Torre, 2016;
  Tutz, 1990). `fit_seq_gdina(responses, q_matrix)` fits ordered polytomous cognitive
  diagnosis by the sequential (continuation-ratio) model: each ordered *step*
  `k in 1..=M_i` of item `i` has a continuation probability `s_ik(l) = P(X_i >= k | X_i
  >= k-1, reduced class l)` that is a saturated G-DINA over the item's `2^{K_i}` reduced
  attribute classes, and the category probabilities are the sequential decomposition
  `P(X_i = k | l) = (prod_{v<=k} s_iv(l))(1 - s_{i,k+1}(l))` with the stop sentinel
  `s_{i,M_i+1} = 0` (top category has no trailing factor — never eps-clamped, so its
  probability carries no spurious bias). Because the sequential likelihood factorizes
  into independent per-step Bernoullis on the at-risk set, the M-step is the closed-form
  saturated ratio `s_ik(l) = R_ik(l)/I_ik(l)` with `R` = expected count reaching category
  `>= k` and `I` = expected count reaching `>= k-1` — exactly `fit_gdina`'s saturated step
  on continuation counts. The population is a free profile distribution `pi_c`; `M_i` is
  derived as each item's maximum observed category (an item stuck at category 0 is
  rejected; a zero-frequency *interior* category is fine — it just means `s_{i,k+1} ~ 1`).
  With one step per item (binary data) it reduces to `fit_gdina` **bit-for-bit** (shared
  monotone init, identical E-step logprobs and closed-form ratio; a regression test
  asserts the whole loglik trace and step probs agree to `< 1e-12`). Deterministic anchors
  pin the sequential core with no Monte-Carlo noise: the category-probability identity
  (`P(0)=1-a, P(1)=a(1-b), P(2)=a*b`, non-centered) and the at-risk-count identity
  (responses `{0,1,1,2}` -> `s_1 = 3/4, s_2 = 1/3`, exercising the `{>=k}/{>=k-1}`
  denominator). A 500-replication Monte-Carlo (K=3, mix of M=2/M=3 items, N=2500, under
  BOTH a normal and a right-skew higher-order attribute distribution) recovers the model
  with category-probability RMSE ~0.020, at-risk-mass-weighted step RMSE ~0.020, and
  attribute-classification agreement ~0.97 — essentially identical across the normal and
  skew conditions, because the free `pi_c` nests the higher-order-implied distribution
  (no prior misspecification). **Scope:** this is the *shared item-level Q-vector*
  sequential G-DINA — every step of an item is a saturated G-DINA over the SAME required
  attributes; it is a restriction of Ma & de la Torre's general per-step (`q_ik`) model,
  whose step-distinct attribute requirements are a deferred non-goal (supply each item's
  Q-vector as the union of its steps' attributes). Compute lives in
  `mlsirm_core::cdm::fit_seq_gdina` (reuses `reduce_class`, the profile-grid posterior,
  and the saturated closed-form ratio); exposed to Python as `fit_seq_gdina` with the
  `SeqGdinaFit` wrapper (`item_step_prob` / `item_cat_prob` ragged accessors).
- **Per-step-Q sequential G-DINA — the full restricted-Q model** (Ma & de la Torre,
  2016). `fit_seq_gdina_qr(responses, step_q, n_steps)` lifts the restriction above: each
  ordered *step* `k` of item `i` carries its OWN attribute requirement `q_ik` (the paper's
  headline generality — step 1 may need attribute A, step 2 need A AND B), supplied as a
  row-major `(sum_i M_i) x K` restricted Q-matrix `Q_r`. The sequential factorization is
  unchanged, so each step is still an independent saturated Bernoulli on its at-risk set
  and the closed-form ratio `s = R/I` is still the exact complete-data MLE — but now each
  step's success is a saturated G-DINA over ITS OWN `2^{|q_ik|}` reduced classes. **Storage
  is union-class-indexed and lossless:** response probabilities depend only on the item's
  UNION `u_i = OR_k q_ik`, so the E-step posterior and the category probabilities are
  indexed by the `2^{|u_i|}` union reduced class (no `N x 2^K` materialization), while each
  step's own reduced class is computed DIRECTLY from the full profile `c` via
  `reduce_class(c, q_ik)` — never a union-mask AND, which would silently mis-gather the
  renumbered set bits. Step probabilities are stored step-row-major (`spo` over `sum_i M_i`
  rows, width `2^{|q_ik|}` each; `step_off[i]` per item, `step_kq[g] = |q_ik|`), category
  probabilities item-major over the union class. **Reduction guard:** giving every step of
  an item the item's Q reproduces `fit_seq_gdina` BIT-EXACTLY (layout-aware cell compare of
  the transposed step tables plus direct compare of the class-major category probs and the
  whole loglik trace — difference exactly `0`). A structural anchor (step 1 `q={A}`, step 2
  `q={A,B}`) asserts the per-step block widths are `2` and `4` (not one collapsed union
  block) and `n_parameters` reflects the per-step widths — a discrimination value recovery
  alone cannot make, since an over-collapse to the union would still fit — while recovering
  a large step-2 B-contrast (`s_2(A1,B0)=0.20` vs `s_2(A1,B1)=0.80`, gap >= 0.4) that the
  shared-Q model cannot represent. `validate` rejects an all-zero step row (a step measuring
  nothing), an attribute required by no step (all-zero union column), and `n_steps[i]` not
  equal to both the declared step count and the maximum observed category, with checked
  `(sum_i M_i) * K` and `2^{|u_i|}` allocations and the same `K` cap. A 500-replication
  Monte-Carlo (K=3, step-distinct M=2/M=3 items plus single-attribute M=1 identification
  items pinning each dimension, N, under BOTH a normal and a right-skew higher-order
  attribute distribution) recovers the model with at-risk-mass-weighted step-probability
  RMSE ~0.017, category-probability RMSE ~0.018, and attribute-classification agreement
  ~0.972 — essentially identical across the normal and skew conditions (the free `pi_c`
  nests the higher-order-implied distribution), 100% convergence with every replication
  finite and on the simplex. Compute lives in `mlsirm_core::cdm::fit_seq_gdina_qr`; exposed
  to Python as `fit_seq_gdina_qr` with the `SeqGdinaQrFit` wrapper (`item_step_prob` ragged
  accessor over the per-step layout). The shared-Q `fit_seq_gdina` is retained as the
  convenience special case.
- **Higher-order G-DINA** (de la Torre & Douglas, 2004; de la Torre, 2011).
  `fit_ho_gdina(responses, q_matrix)` fits the saturated G-DINA item model (each
  item's reduced attribute-mastery classes get a free success probability) under a
  *higher-order structural attribute prior*: a continuous trait `theta ~ N(0,1)`
  drives mastery, `P(alpha_k=1 | theta) = sigmoid(a_k theta + d_k)`, with attributes
  conditionally independent given the trait. It generalizes `fit_ho_cdm` (which
  restricts the item model to DINA slip/guess) and constrains `fit_gdina`'s free
  class distribution to the `2K`-parameter structured family. Estimated by
  marginal-ML EM over the joint `(alpha, theta)` grid: because the item response is
  conditionally independent of the trait given the attributes, the saturated item
  M-step `p_il = R_il/I_il` marginalizes the trait out exactly (reusing `fit_gdina`'s
  closed form), and the structural step is `K` independent 2PL calibrations of
  attribute mastery on the trait (reusing `fit_ho_cdm`'s Newton). The higher-order
  parameters are identified for `K >= 3`. Validated by a non-trivial anchor (a free
  saturated fit of DINA-patterned data recovers the DINA identity-link `delta`
  *and* the higher-order parameters), an independent-attribute pi-recovery check, and
  a 500-replication Monte-Carlo study (K=3, N=1500) — the saturated item
  probabilities recover with mass-weighted RMSE ~0.02 and attribute agreement > 0.9
  under both a normal and a skewed trait distribution. Extends `mlsirm_core::cdm`
  (reuses `reduce_class`, `mobius_inverse_inplace`, `newton_attr_2pl`,
  `ho_pi_from_params`). Exposed to Python through PyO3 as `fit_ho_gdina` with the
  `HoGdinaFit` wrapper.

- **Rating Scale Model** (Andrich, 1978). `fit_rsm(responses)` fits the Rasch-family
  polytomous model for items on a common rating scale (e.g. Likert): every item has
  its own location `delta_i`, but the `K-1` category thresholds `tau_k` are *shared
  across all items* — `ln[P(X=k)/P(X=k-1)] = theta - delta_i - tau_k`, `theta ~
  N(0,1)`. This is a constrained partial-credit model (the PCM/GPCM in `poly.rs` /
  `mixed.rs` have item-specific thresholds); at `K=2` it reduces exactly to the Rasch
  model. Implemented as the GPCM cell with slope 1 and the structured intercept
  `-k*delta_i - sum_{m<=k} tau_m` (reusing `poly::gpcm_logprobs`), fit by marginal-ML
  EM with a monotone ECM M-step: a per-item Newton for the locations, then a joint
  Newton for the shared thresholds aggregated over items — both with a backtracking
  line search that guarantees the marginal likelihood ascends — followed by
  re-centering the thresholds to sum to zero (the model is invariant under
  `tau -> tau - c`, `delta -> delta - c`). A 500-replication Monte-Carlo study (J=12,
  K=5, N=1000) recovers the item locations and the shared thresholds tightly and the
  trait with correlation > 0.85 under both a normal and a skewed trait distribution.
  New `mlsirm_core::rsm` module; exposed to Python through PyO3 as `fit_rsm` with the
  `RsmFit` wrapper.

- **Continuous Response Model** (Samejima, 1973) — the library's first estimator
  for a *continuous* bounded response (all other models are binary, polytomous,
  response-time, or cognitive-diagnosis). `fit_crm(responses)` fits Samejima's CRM,
  the limit of the graded response model as the number of ordered categories grows
  without bound. Operationally (Wang & Zeng, 1998), the logit of a response
  `Z in (0,1)` is conditionally normal and linear in the trait:
  `logit(Z_ij) | theta_j ~ N(a_i theta_j + d_i, sigma_i^2)`, `theta ~ N(0,1)`. The
  working `(slope a_i, intercept d_i, residual sd sigma_i)` map to the classic
  `(discrimination alpha_i = a_i/sigma_i, difficulty b_i = -d_i/a_i, scale
  gamma_i = a_i)`, all reported. Estimated by marginal-ML EM over a Gauss-Hermite
  trait grid with a **closed-form** weighted-least-squares item M-step (regress the
  transformed response on the trait under the posterior, then the residual
  variance) — the exact profile MLE, no Newton iteration. Continuous responses are
  information-rich, so a 500-replication Monte-Carlo study (J=15, N=500) recovers
  the item parameters tightly and the trait with correlation > 0.9 under both a
  normal and a skewed trait distribution. New `mlsirm_core::crm` module (reuses the
  `quadrature::gh_rule` grid); exposed to Python through PyO3 as `fit_crm` with the
  `CrmFit` wrapper. The `Z -> logit` Jacobian is a data-only constant, so the
  reported log-likelihood is in the transformed metric.

- **Higher-order structured attribute prior for cognitive diagnosis** (de la Torre
  & Douglas, 2004). `fit_ho_cdm(responses, q_matrix, model="dina"|"dino")` fits a
  DINA/DINO model whose `2^K` attribute-class distribution, instead of being free
  (as in `fit_cdm`), is *structured* by a continuous higher-order trait
  `theta ~ N(0,1)`: `P(alpha_k=1 | theta) = sigmoid(a_k theta + d_k)` with attributes
  conditionally independent given the trait. This replaces the `2^K - 1` free class
  probabilities with `2K` interpretable attribute parameters (slope `a_k`,
  intercept `d_k`). Estimated by marginal-ML EM over the joint `(alpha, theta)` grid:
  the item slip/guess M-step is unchanged, and the population update becomes `K`
  independent 2PL calibrations of attribute mastery on the trait (reusing the
  `fit_mmle_2pl` Newton with expected node counts). The implied class distribution,
  per-person trait EAP, MAP profile, and marginal attribute mastery are returned.
  The observed-data likelihood depends on `(a_k, d_k)` only through the implied class
  distribution, so the higher-order parameters are a genuine, identified restriction
  only for `K >= 3` (at `K <= 2` only the class distribution and the attribute
  classification are identified); `attr_slope` is anchored non-negative. A
  500-replication Monte-Carlo study (higher-order DINA, K=3, N=1000) recovers the
  attribute parameters and classification under both a correctly-specified normal
  trait and a mis-specified skewed trait. Extends `mlsirm_core::cdm` — reuses the
  DINA gate, `update_item`, and `mmle::GH_NODES`/`GH_WEIGHTS`. Exposed to Python
  through PyO3 as `fit_ho_cdm` with the `HoCdmFit` wrapper.

- **Item-level cognitive-diagnosis model selection by the Wald test** (de la
  Torre, 2011). `gdina_wald_selection(responses, q_matrix, alpha=0.05)` tests, for
  each item, whether the saturated G-DINA can be replaced by a more parsimonious
  reduced model. The candidates are exact *linear restrictions* of the
  identity-link parameters `delta = M^{-1} P` (`P` the reduced-class success
  probabilities): **DINA** (conjunctive — only the intercept and the top-order
  interaction free), **DINO** (disjunctive — the non-intercept coordinates tied
  onto one line `delta_S = (-1)^{|S|+1} Delta`, a general non-coordinate
  restriction), **A-CDM** (additive on the identity link — all interaction
  coordinates zero), **LLM** (linear logistic model — additive on the *logit* link),
  and **R-RUM** (reduced reparameterized unified model — additive on the *log* link).
  The Wald statistic `W = (R delta)' (R Sigma_delta R')^{-1} (R delta) ~ chi^2(df)`
  restricts the identity-link `delta = M^{-1} P` for DINA/DINO/A-CDM and the
  transformed `delta^h = M^{-1} h(P)` for LLM (`h = logit`) and R-RUM (`h = log`).
  For the identity link `Sigma_delta = M^{-1} Var(P) M^{-T}` with
  `Var(P_l) = P_l(1-P_l)/I_l`; for a transformed link the first-order delta method
  gives `Var(h(P_l)) = h'(P_l)^2 Var(P_l)` (LLM `1/(I_l P_l(1-P_l))`, R-RUM
  `(1-P_l)/(I_l P_l)`), sharing the same Möbius sandwich. All three covariances (and
  the two transformed deltas) accumulate in one pass over the shared Möbius columns
  `c_l = M^{-1} e_l` (reusing `mobius_inverse_inplace`); the expected reduced-class
  counts `I_l` come from one posterior pass. Per item the fewest-parameter model not
  rejected at `alpha` is selected (DINA and DINO cost two parameters; A-CDM, LLM and
  R-RUM each cost `1 + K`, so ties are broken by the larger p-value), else the
  saturated G-DINA. The covariance uses complete-data (expected) rather than
  observed information, so the test is mildly liberal — a 500-replication
  Monte-Carlo study (K=2, N=3000, strong attribute identification) confirms Type I
  error near nominal under both uniform and correlated/skew attribute distributions
  (Type I at `alpha=0.05`: DINA/DINO/A-CDM/LLM/R-RUM all within ~0.059–0.083) with
  power 0.98–1.000 against false over-restrictive or wrong-link models — including the
  cross-link cases (A-CDM and R-RUM rejected under LLM truth ~1.000/0.98, LLM rejected
  under R-RUM truth 1.000), verifying the link transform is faithful rather than
  cosmetic. A
  non-centered anchor test drives this home: truths additive on *only* one of the
  three links (identity/logit/log) are each recovered as their own model while the
  other two additive models are rejected. Extends `mlsirm_core::cdm` (reuses
  `fit_gdina`, `reduce_class`, `posterior_row_gdina`, `mobius_inverse_inplace`, and
  `fitstats::chi2_sf`). Exposed to Python through `gdina_wald_selection` /
  `WaldModelSelection` (both generic in the model count, so the two new candidates
  flow through unchanged). Deferred: the incomplete-data (observed-information)
  covariance.

- **Empirical Q-matrix validation by the PVAF method** (de la Torre & Chiu,
  2016). `validate_q_matrix(responses, provisional_q, epsilon=0.95)` checks and
  corrects the attribute-by-item Q-matrix of a cognitive-diagnosis model. Each
  candidate q-vector groups the `2^K` latent attribute classes into masters vs.
  non-masters of its required attributes; the *proportion of variance accounted
  for* is `PVAF(q) = zeta^2(q) / zeta^2_full`, the share of the item's
  across-class success-probability variance that grouping captures. Per item the
  method returns the q-vector with the **fewest** required attributes whose
  `PVAF >= epsilon` — an under-specified provisional q falls short and is
  enlarged, an over-specified one is trimmed because a smaller vector already
  clears the cutoff. The class weights and identified attribute labels come from
  a G-DINA fit under the provisional Q; each item's *saturated* success
  probability over all `2^K` classes is then recovered nonparametrically from
  the fitted posteriors, so a mis-specified item's true dependence is exposed by
  the attributes the *other* items identify (the method assumes the provisional
  Q is mostly correct). Extends `mlsirm_core::cdm` — reuses the G-DINA
  `reduce_class` collapse and posterior pass; the exhaustive q-vector search is
  `O(J * 4^K)`, so `K` is capped at 10. Validated by an anchor (the true Q
  validates to itself), over-/under-specification correction, and a
  500-replication Monte-Carlo Q-recovery study (K=3, J=15, N=1000): under a
  uniform attribute distribution the exact q-vector is recovered for 98.1% of
  items (attribute TPR 0.996, FPR 0.012), and under a correlated/skew
  higher-order distribution for 93.5% (TPR 0.982, FPR 0.035). Exposed to Python
  through PyO3 as `validate_q_matrix` with the `QMatrixValidation` wrapper.
  Deferred: the stepwise Wald item-level model-selection test (de la Torre,
  2011) and sequential/iterative Q-matrix re-estimation.

- **Testlet response model** (Bradlow, Wainer, & Wang, 1999; Wang, Bradlow, &
  Wainer, 2002). `fit_testlet(responses, testlet_id, model="rasch"|"2pl")` models the
  local dependence induced when items share a common stimulus (a reading passage): each
  item in testlet `d` carries a person-specific random effect `gamma_{j,d} ~ N(0,
  sigma^2_d)`, so `P(X=1) = sigmoid(a_i(theta_j - b_i - gamma_{j,d(i)}))`. The per-testlet
  variance `sigma^2_d` is the estimand of interest — a large value flags strong
  within-bundle dependence; all `sigma^2_d = 0` is the ordinary conditional-independence
  2PL/Rasch model. A dedicated estimator (not the general bifactor): because each item
  depends on `theta` and exactly one testlet effect, the marginal likelihood **factors**
  into a `theta`-outer / per-testlet-`gamma`-inner nested Gauss-Hermite quadrature whose
  per-person cost is independent of the number of testlets `D` (vs the bifactor's
  exponential `D`-dimensional grid), and it reports `sigma^2_d` directly rather than only
  per-item loadings. The item M-step reuses `fit_mmle_2pl`'s Newton on the effective node
  `t_g - sigma_d*u_h`; the closed-form variance update `sigma^2_d <- sigma^2_d * mean_j
  E[u_d^2 | y_j]` is accelerated with SQUAREM (Varadhan & Roland, 2008; monotone, with a
  plain-EM fallback) to tame the slow variance-component convergence. Singleton testlets
  (whose variance is non-identified) are pinned to 0. Compute lives in
  `mlsirm_core::testlet::fit_testlet`; the shared Newton and Gauss-Hermite table make the
  `sigma^2 -> 0` case reduce **bit-exactly** to `fit_mmle_2pl` (the reduction anchor,
  asserted `< 1e-12`). Also anchored: a no-spurious-LD check (pure-2PL data recovers
  `sigma^2 ~ 0`), a strong-LD recovery with a log-likelihood gain over the naive 2PL fit,
  singleton pinning, and a monotone-ascent guard. A Bradlow-Wainer-Wang-style
  500-replication Monte-Carlo (Rasch testlet, N=1000, D=4) under normal and skewed
  ability recovers the testlet variances near-unbiasedly (RMSE ~0.093, `|bias| <= 0.007`)
  and the item difficulties (RMSE ~0.09), with every replication converging. Exposed via
  PyO3 as `fit_testlet` with the
  `TestletFit` Python wrapper. (In the 2PL testlet the discrimination `a_i` and the
  testlet SD `sigma_d` both scale the dependence via `a_i*sigma_d` and separate only
  weakly, so the Rasch testlet is the well-identified default.) Deferred: polytomous and
  3PL testlets, covariate/second-order structure, and the original paper's fully-Bayesian
  MCMC estimator.

- **Linear Logistic Test Model (LLTM)** (Fischer, 1973). An *explanatory* Rasch
  model: `fit_lltm(responses, q_design)` decomposes each item's easiness (the package's
  additive sign convention; Fischer difficulty is its negative) into a
  linear combination of `K` basic cognitive-operation parameters through a fixed
  weight matrix `Q` (`b_i = c + Σ_k q_ik·η_k`), rather than estimating `J` free item
  easinesses. With `K << J` parameters it tests whether a small set of cognitive
  operations *explains* the item parameters. Estimated by marginal-ML EM: the
  E-step is the Rasch node posterior over the shared Gauss-Hermite rule; the M-step is
  a `K`-dimensional chain-rule Newton — the per-item Rasch easiness gradient/Hessian
  aggregated through the design (`g_η = Qᵀg_b`, `H_η = Qᵀ diag(h_b) Q + ridge`, solved
  with the shared dense `solve_small`). A free grand-mean easiness intercept is fit by
  default. The classic likelihood-ratio test of LLTM vs the saturated Rasch model
  (`2·(ll_Rasch − ll_LLTM) ~ χ²(J − K − 1)`) is computed inline (the Rasch reference is
  the same engine run with `Q = I`). **Identification is validated, not assumed**: the
  effective design (including the intercept column) must have full column rank for `η`
  to be identified, so a rank-deficient `Q` (e.g. one whose rows sum to a constant,
  colliding with the intercept) is rejected rather than papered over by the Newton
  ridge. Compute lives in `mlsirm_core::lltm::fit_lltm`; because the M-step reuses
  `mmle`'s Rasch Newton and Gauss-Hermite table, the `Q = I` case reduces
  **bit-exactly** to a Rasch fit — anchored two ways: a single M-step is bit-identical
  (`==`) to `J` independent per-item Rasch Newton steps, and a full `Q = I` fit matches
  a single-class Rasch mixture fit to `< 1e-10`. A 500-replication Monte-Carlo
  (J=30, K=5, N=1500) under normal and skewed ability recovers the basic parameters
  (RMSE/bias) and induced easinesses, and validates the LR test (Type I when the
  restriction holds, power when it is violated off-model). Exposed via PyO3 as
  `fit_lltm` with the `LltmFit` Python wrapper. This is the marginal-ML / `N(0,1)`
  operationalization of Fischer's conditional-ML LLTM. It is a repository-specific
  estimator choice, and finite-sample equality with Fischer's conditional-ML item
  estimates is not asserted. Deferred: conditional-ML estimation, LLTM for 2PL/polytomous
  models, and random-weights / LLRA extensions.

- **Mixed Rasch / mixture IRT** (Rost, 1990; Rost & von Davier, 1995). A new
  paradigm for unobserved population heterogeneity: `fit_mixture(responses,
  n_classes, model="rasch"|"2pl")` models the population as a mixture of `C` latent
  classes, each with its OWN item parameters and a mixing weight `pi_c`, detecting
  qualitatively different response strategies a single-class model cannot represent.
  Within a class, responses follow a Rasch (discrimination fixed at 1) or 2PL model
  with `theta ~ N(0,1)`, estimated by marginal-ML EM: the E-step forms the joint
  posterior over (class, ability node) via one max-shift log-sum-exp over the `C·Q`
  Gauss-Hermite grid; the per-class item M-step reuses the exact penalized Newton
  step of `fit_mmle_2pl` (weighted by the class responsibility); the mixing weights
  update to the mean posterior class membership. Because the mixture likelihood is
  multimodal, `n_starts > 1` runs random restarts (start 0 is a deterministic warm
  start) and keeps the highest-likelihood fit; classes are returned in a canonical
  order (mixing weight descending, ties by mean difficulty ascending) to tame label
  switching. Compute lives in `mlsirm_core::mixture::fit_mixture`; the shared Newton /
  Gauss-Hermite table with `fit_mmle_2pl` makes the `C = 1` case reduce **bit-exactly**
  to the verified single-class 2PL estimator — the reduction anchor, asserted to
  `< 1e-12`. Also anchored: a two-class difficulty-reversal recovery (the canonical
  Rost two-strategy structure), permutation-matched, plus a monotone-ascent guard. A
  500-replication Monte-Carlo (C=2, J=15, N=1500, reversal truth) under normal and
  skewed ability recovers the class difficulties (permutation-matched RMSE), mixing
  proportions, and class membership (MAP accuracy + label-invariant Adjusted Rand
  Index; Hubert & Arabie, 1985). Exposed via PyO3 as `fit_mixture` with the
  `MixtureFit` Python wrapper. This repository combines Rost's latent-class structure
  with a fixed-standard-normal, Bock-Aitkin marginal-ML EM estimator. It differs from
  the conditional-ML estimators in Rost (1990) and psychomix (Frick et al., 2012), so
  no exact finite-sample item-contrast equivalence is claimed. Deferred: free per-class
  ability variance, automatic model selection
  over `C` (AIC/BIC/ICL from the returned `n_parameters`/`loglik_trace`), and
  concomitant-variable mixing.

- **Generalized DINA (G-DINA), the saturated cognitive-diagnosis framework**
  (de la Torre, 2011). `fit_gdina(responses, q_matrix)` fits the general model of
  which DINA, DINO, A-CDM, LLM, and R-RUM are constrained special cases. For an
  item requiring `K_i` attributes, each of its `2^{K_i}` *reduced* attribute-mastery
  classes gets a **free** success probability `p_il = P(X_i = 1 | reduced class l)`,
  estimated by marginal-ML EM over the `2^K` profiles. The E-step reuses the DINA
  profile-grid posterior; the closed-form saturated M-step is
  `p_il = R_il / I_il` (expected correct / expected total in reduced class `l`) —
  exactly DINA's two-cell slip/guess step generalized to `2^{K_i}` cells (de la
  Torre, 2011, Eq. 10). The identity-link parameters `item_delta` (intercept, main
  effects, all interactions) are recovered from the fitted probabilities by an
  in-place signed subset Möbius transform `delta = M^{-1} p` (no matrix inverse), so
  the constrained submodels are readable off the `delta` pattern — DINA leaves only
  the intercept and the highest-order interaction nonzero; A-CDM zeroes the
  interactions. Item parameters are stored ragged (CSR: `item_off` + flat
  `item_prob`/`item_delta`) since `2^{K_i}` varies per item; the box constraint
  `0 <= p_il <= 1` holds for free (`0 <= R_il <= I_il`). The saturated estimator is
  otherwise order-unconstrained: Q-matrix identifiability does not make the
  all-mastered class largest, and the separate Hong, Chang, and Tsai (2016)
  subset-lattice order restriction is not implemented.
  Compute lives in `mlsirm_core::cdm::fit_gdina`, extending the DINA module without
  touching the shipped DINA core; exposed via PyO3 as `fit_gdina` with the `GdinaFit`
  Python wrapper. Correctness is anchored by a brute-force likelihood identity
  (log-space path == naive enumeration to `1e-12`), a **DINA-reduction crux anchor**
  (DINA-generated data recovers `p_il = g_i` for every non-top class and `1 - s_i`
  at the top, with the exact DINA `delta` pattern), a DINO-reduction anchor, an
  A-CDM additivity anchor (fitted interactions negligible relative to main effects),
  a Möbius round-trip identity, an exhaustive `reduce_class` bit-packing check, and a
  deterministic limit. A de la Torre (2011)-style 500-replication Monte-Carlo (K=5,
  J=30, N=1000) with a stochastic higher-order attribute distribution (de la Torre &
  Douglas, 2004) under normal and skewed abilities recovers `p_il` (mass-weighted
  RMSE) and attribute classification accuracy. Deferred: LLM/R-RUM logit/log-link
  submodels, item-level model-selection Wald tests, Q-matrix validation, and full
  subset-lattice isotonic monotonicity (Hong et al., 2016).

- **Cognitive diagnosis models: DINA and DINO** (Junker & Sijtsma, 2001; de la
  Torre, 2009; Templin & Henson, 2006). A new discrete-attribute paradigm
  alongside the continuous-trait family: `fit_cdm(responses, q_matrix,
  model="dina"|"dino")` classifies each respondent's binary attribute-mastery
  profile `alpha in {0,1}^K` against a Q-matrix of item-attribute requirements.
  The ideal response is the conjunctive AND gate `eta = prod_k alpha_k^{q_k}`
  (DINA — mastery of all required attributes) or the disjunctive OR gate
  `eta = 1 - prod_k (1-alpha_k)^{q_k}` (DINO — any required attribute), and the
  observed response adds a per-item slip `s_i = P(X=0|mastered)` and guess
  `g_i = P(X=1|not mastered)`, `P(X=1|alpha) = (1-s_i)^{eta}(g_i)^{1-eta}`.
  Estimation is marginal-ML EM over the `2^K` profiles with a free profile
  distribution: the E-step posterior is accumulated over the discrete profile
  grid (a bitwise gate test replaces the continuous quadrature), the item M-step
  is **closed form** (`s_i = 1 - R1_i/I1_i` = expected fraction of masters
  answering wrong; `g_i = R0_i/I0_i` = non-masters answering right; de la Torre,
  2009, Eqs. 9-10), and the population step is a mean of the posteriors. The
  monotonicity/identification constraint `1 - s_i > g_i` is enforced by the exact
  constrained boundary maximiser; missing cells are dropped under MAR. Persons
  are classified by the posterior-mode profile (`map_profile`) and marginal
  attribute-mastery probabilities (`attr_prob`, attribute EAP). All compute runs
  in the Rust core (`mlsirm_core::cdm::fit_cdm`) with the `2^K` profile grid
  bit-encoded (no `N*L` storage; streaming E-step); DINA and DINO share one
  estimator differing only in the one-line gate mask. Correctness is anchored by
  a brute-force likelihood identity (log-space path == naive enumeration to
  `1e-12`), a deterministic `s=g=0` limit (exact pattern recovery), a
  DINA==DINO gate-equivalence identity on single-attribute items, and a K=1
  reduction to a 2-class latent-class model. A de la Torre (2009)-style
  500-replication Monte-Carlo (K=5, J=30, N=1000) recovers slip/guess with mean
  RMSE 0.013-0.024 and negligible bias (`|bias| < 3e-4`) and attains attribute
  classification agreement 0.99 (s=g=0.1) / 0.95 (s=g=0.2), pattern-wise 0.96 /
  0.76. Deferred: the general G-DINA/saturated CDM, Q-matrix estimation, and
  structured (higher-order) attribute priors.

- **Polytomous response models (GRM / GPCM), unidimensional.** A complete
  fit -> score -> information subsystem: `fit_polytomous(responses, n_cat,
  model="grm"|"gpcm")` fits the graded response model (Samejima; the default)
  or the generalized partial credit model (Muraki) by Bock-Aitkin marginal-EM;
  `score_polytomous(responses, fit)` returns EAP trait scores and posterior
  SDs; `information_polytomous(fit, theta)` returns item and test Fisher
  information curves. `NaN` responses are treated as missing and marginalized
  out of each person's likelihood and posterior. All numerical work — the category cells, the residual
  M-step gradient, the Newton item update, the EAP reduction, and the
  information — runs in the Rust core (`mlsirm_core::poly`:
  `grm_logprobs`/`gpcm_logprobs` + `*_node_gradient` + `fit_poly_unidim` +
  `score_poly_eap` + `poly_item_information`), exposed via PyO3; the NumPy
  `category_logprobs`/`grm_category_logprobs`/`gpcm_node_gradient`/
  `fit_gpcm_numpy` are parity references held to `<= 1e-12` (both cells) /
  recovery agreement (fitter). GRM is
  chosen as the identification-clean default for the latent-space family — the
  single interaction term enters every cumulative logit as a shared shift, with
  no forced category scaling (design rationale and literature basis in
  `docs/papers/gpcm-nominal-design-spec.md`). The latent-space polytomous
  extension (the same cell inside the marginal `(theta, xi)` quadrature) is the
  next milestone.

- **Polytomous computerized adaptive testing** (Dodd, De Ayala & Koch, 1995).
  `cat_simulate_polytomous(true_theta, fit)` simulates an adaptive test over a
  fitted GRM/GPCM bank: items are selected by maximum Fisher information at the
  running EAP trait, responses are generated at the true trait, and the trait +
  posterior SD are re-estimated after each item, stopping at an SE threshold (or
  a fixed length). Returns per-simulee `theta_eap`, `theta_sd`, and `n_used`.
  Compute in Rust (`mlsirm_core::poly::poly_cat_simulate`, plus
  `poly_cat_next_item`), composing the existing item information and EAP scoring.
  Validated by a 500-simulee Monte-Carlo: a variable-length CAT recovers the
  trait to RMSE 0.29 (normal) / 0.33 (skew) using ~9.7 of 40 bank items, and at
  a fixed length of 12 maximum-information selection beats random (RMSE 0.27 vs
  0.33 normal; 0.30 vs 0.40 skew).

- **Polytomous person fit** (Drasgow, Levine & Williams, 1985; Snijders, 2001).
  `person_fit_polytomous(responses, fit)` returns the standardized
  log-likelihood `l_z` and its estimated-trait correction `l_z*` (per person,
  at the EAP trait) plus `theta_eap` and a boolean `flagged`, for a fitted
  GRM/GPCM — the ordered-category generalization of the binary l_z. Compute in
  Rust (`mlsirm_core::poly::poly_person_fit`), reusing the poly cells with a
  central-difference trait score. Validated by an exact reduction to the binary
  `person_fit` l_z at `n_cat = 2` (`<1e-6`) and a 500-replication Monte-Carlo:
  under model respondents `l_z*` is ~N(0,1) (mean −0.15, sd 1.04, Type I 0.08
  at a 20-item test), and inconsistent responders are flagged with power 0.86.

- **Nominal categories model** (Bock, 1972; Thissen, Cai & Bock, 2010).
  `fit_nominal_polytomous(responses, n_cat)` fits the unidimensional nominal
  model `P(Y=k|θ) = softmax_k(a_k·θ + c_k)` with a free scoring function `a_k`
  and intercept `c_k` per category, identified by `a_0 = c_0 = 0` and
  `θ ~ N(0,1)`, returning a `NominalFit` (`scores`, `intercepts`, `loglik`).
  The generalized partial credit model is the special case `a_k = a·k`, so the
  nominal model nests it. Compute in Rust (`mlsirm_core::poly::fit_nominal`),
  reusing the softmax cell and its residual gradient. The parameterization and
  identification were adversarially verified against the source chapter.
  Validated by the GPCM nesting (loglik ≥ the GPCM fit, recovered scores linear
  in `k`) and a 500-replication recovery Monte-Carlo (per-item sign alignment):
  under a matched `N(0,1)` ability the score RMSE is 0.15 with |bias| 0.01
  (near-unbiased), degrading to RMSE 0.44 / |bias| 0.39 under a skewed
  population.

- **Polytomous item-pair local dependence** (Chen & Thissen, 1997; Liu &
  Maydeu-Olivares, 2013). `local_dependence_polytomous(responses, fit)` returns,
  for every item pair of a fitted GRM/GPCM, the Pearson `X²` and likelihood-ratio
  `G²` comparing the observed `K×K` contingency table to the model-implied joint
  under local independence, with `df = (K-1)²`, the χ² p-value, Cramér's V, and
  the largest standardized cell residual — the ordered-category generalization
  of the binary pairwise χ² and the pair-level complement to item-level S-X² and
  test-level M2. Compute in Rust (`mlsirm_core::fitstats::poly_local_dependence`).
  Validated by a deterministic K=2 reduction to a from-scratch 2×2 χ² and a
  500-replication Monte-Carlo at fitted parameters: locally-independent pairs are
  calibrated (X²/df = 0.84, Type I 0.03 — conservative, as the papers note),
  while an injected 2-item testlet is localized to that pair (X²/df = 10.9, power
  1.00).

- **Polytomous IRT likelihood-ratio DIF** (Thissen, Steinberg & Wainer, 1993;
  Woehr & Meriac, 2010). `dif_polytomous(responses, group_id, n_cat)` runs a
  two-group DIF sweep for GRM/GPCM items: it fits a *compact* model (all items
  group-invariant) once, then per studied item an *augmented* model (that item's
  parameters freed per group) with every other item as the anchor, and refers
  `LR = 2·Δloglik` to `χ²((n_groups−1)·n_cat)`. Each non-reference group's latent
  distribution `N(μ_g, σ_g²)` is estimated in **both** models (group 0 pinned to
  `N(0,1)`), so genuine ability differences between groups (impact) are absorbed
  rather than mistaken for DIF. Returns per-item `lr`, `df`, `p_value`,
  `flagged_bh` (Benjamini-Hochberg FDR), and `effect_size` (the across-group
  range of the item's mean category location). Compute in Rust
  (`mlsirm_core::poly::fit_poly_multigroup` — a Bock-Zimowski multi-group
  marginal EM whose per-item M-step reuses the single-group Newton step on each
  group's nodes/expected-counts stacked, the concatenation being exactly the
  Bock-Zimowski pooling — driving `poly_dif_sweep`). Validated by a 500-rep
  Monte-Carlo with impact (focal `θ~N(0.5, 1.2²)`), two-group GPCM, `K=3`:
  under no DIF the test is calibrated (Type I 0.042, `mean(LR)=2.92≈df=3`), an
  injected uniform difficulty shift is detected with power 0.996 and a
  non-uniform slope difference with power 0.920, while a skewed focal population
  inflates Type I only mildly (0.057); a structural check confirms the augmented
  fit never falls below the compact one and recovers the focal `μ, σ`.

- **Response-time person fit** (van der Linden & Guo, 2008; Sinharay, 2018).
  `rt_person_fit` flags aberrant response-time patterns — rapid guessing, item
  preknowledge — under a fitted lognormal RT model. It profiles each person's speed
  by ML, so the sum of squared standardized log-time residuals
  `W_j = sum_i [alpha_i (ln T_ij - (beta_i - tau_hat_j))]^2` is *exactly*
  `chi2(n_j - 1)` (an orthogonal-projection identity — the estimated-speed
  correction is a clean loss of one degree of freedom, the RT analogue of `l_z*`,
  with no asymptotic drift). It returns the aggregate `W`/p-value, a Wilson-Hilferty
  standardized `l_t`, and per-item studentized residuals plus one-sided too-fast
  flags. It detects speed *inconsistency across items*, not a uniform speed level
  (the profile absorbs it). Compute in Rust (`rt::rt_person_fit`, reusing
  `fitstats::chi2_sf`); exposed via PyO3 and Python. Validated by an exact identity
  anchor (at true parameters the residuals are `N(0,1)` and `W` is `chi2(n)` with
  known speed, `chi2(n-1)` once profiled, to within Monte-Carlo error) and a
  500-replication Monte-Carlo: Type I sits on nominal (0.05, exact — no
  finite-length conservatism), rapid-guessing and preknowledge responders are
  detected with power ~1.0 under both normal and skew speed, the flag is robust to
  the speed-distribution shape (it conditions on within-item residuals), and the
  tampered items are recalled at ~99%. Deferred: an EAP-plug-in mode (statistically
  inferior — it mis-calibrates the chi-square) and multivariate RT aberrance.

- **Joint speed-accuracy hierarchical model** (van der Linden, 2007, Level 2). A
  new `mlsirm_core::rt_joint` module and the public `fit_speed_accuracy` — the
  person-level layer that ties ability `theta` (from an accuracy 2PL model) to
  speed `tau` (from the lognormal RT model) through a bivariate-normal person
  distribution `(theta, tau) ~ N2(0, [[1, rho*sigma_tau], [rho*sigma_tau,
  sigma_tau^2]])`, with the accuracy responses and log-times conditionally
  independent given `(theta, tau)`. The headline output is `rho`, the ability-speed
  correlation. This is the two-stage estimator: item parameters are held fixed and
  the person covariance `(rho, sigma_tau)` is estimated by marginal ML over a 2-D
  Gauss-Hermite grid built by Cholesky-mapping the standard nodes through
  `Sigma_P`, with an exact constrained EM M-step (`c = S12/S11`,
  `v = S22 - S12^2(S11-1)/S11^2`). The reported `rho` is the consistent marginal-ML
  correlation, not the shrinkage-attenuated correlation of the two separate EAPs.
  Compute in Rust (`rt_joint::fit_speed_accuracy_covariance`); exposed via PyO3 and
  Python. Validated by an exact identity anchor (at `rho = 0` the 2-D grid
  log-likelihood factorizes into the sum of the two 1-D grids to `< 1e-10`), a
  reduction anchor (true independence returns `rho ~ 0`), monotone EM, and a
  500-replication Monte-Carlo recovering `rho in {0, 0.5, -0.5}` with essentially
  zero bias (bias `< 0.001`, RMSE ~0.03-0.04) and `sigma_tau` to RMSE ~0.008.
  Deferred: the one-step full-information MMLE, 3PL guessing, and item-parameter-
  uncertainty propagation into SE(rho).

- **Lognormal response-time model** (van der Linden, 2007). A new
  `mlsirm_core::rt` module and the public `fit_response_times` — the speed-side
  analogue of the 2PL for item response *times*, opening a response-time modality
  alongside the accuracy models. For person `j` (latent speed `tau_j`) and item
  `i` (time intensity `beta_i`, time discrimination `alpha_i`),
  `ln(T_ij) ~ Normal(beta_i - tau_j, 1/alpha_i^2)`; item parameters and the speed
  SD are estimated by marginal-ML EM with `tau ~ Normal(0, sigma_tau^2)`, and speed
  is scored by EAP. Because the model is conditionally Gaussian with a unit loading
  on `tau`, the speed posterior, marginal likelihood, and EAP are all *exact closed
  forms* (matrix-determinant / Sherman-Morrison), so the estimator needs neither
  quadrature nor a line search — the EM is exact `O(nnz)` coordinate ascent. The
  log-time metric identifies the speed scale (so `sigma_tau` is estimated, not
  fixed) and only the location is pinned (`mu_tau = 0`). Compute in Rust; exposed
  via PyO3 and Python; missing/non-positive times are marginalized per person.
  Validated by an exact identity anchor (the closed-form marginal log-likelihood
  equals a dense multivariate-normal log-pdf to `< 1e-9`), a reduction anchor
  (`sigma_tau -> 0` collapses to the per-item lognormal MLE), and a 500-replication
  Monte-Carlo: under both normal and a *misspecified* skew speed population the item
  parameters stay essentially unbiased (RMSE `alpha` 0.067 / `beta` 0.027, bias
  `beta` -0.0001 under skew) with speed recovered at corr 0.92, demonstrating that
  the level-1 RT item parameters are estimable independently of the speed
  distribution's shape. Deferred: the joint speed-accuracy hierarchical layer,
  Louis-standard-error information, and RT bank linking.

- **Standard errors of equating** (Kolen & Brennan, 2014, ch. 7; Efron &
  Tibshirani, 1993). `equating_standard_errors` reports the per-score-point
  sampling error of the equated score for the equivalent-groups design, by two
  routes. The nonparametric **bootstrap** (`route="bootstrap"`) resamples
  examinees per group independently with replacement at the observed sample sizes,
  re-equates each of `n_boot` replicates through the existing equating code, and
  returns the per-score bootstrap SD and a percentile confidence interval — it
  works for every method including equipercentile, which has no simple analytic
  SEE. The **delta-method** (`route="analytic"`) returns the closed-form
  normal-theory SE for mean equating (`sigma_x^2/n_x + sigma_y^2/n_y`, constant in
  `x`) and linear equating (`sigma_y^2 (1 + z^2/2)(1/n_x + 1/n_y)`,
  `z = (x-mu_x)/sigma_x`). Compute in Rust (`equating::bootstrap_see` /
  `analytic_see`); exposed via PyO3 and Python. Validated by the analytic-Linear
  agreeing with the bootstrap-Linear SEE within Monte-Carlo tolerance, the Mean
  SEE being constant, a `1/sqrt(N)` shrink and seed-determinism check, and a
  500-replication Monte-Carlo confirming the bootstrap SE recovers the *true*
  sampling SD of `e_Y(x)` (from an outer fresh-sample Monte-Carlo) — interior
  ratio in [0.95, 1.08] for equipercentile. Deferred: NEAT bootstrap SEE, analytic
  equipercentile/kernel SEE.

- **Tucker & Levine linear NEAT equating** (Kolen & Brennan, 2014, §4.3–4.4;
  Brennan, 2006). `equate_neat_linear` adds the linear observed-score methods for
  the common-item non-equivalent-groups design, alongside the existing chained /
  frequency-estimation equipercentile NEAT. Each forms synthetic-population
  moments of the two forms (weighted by `w1`) from a group total-on-anchor slope
  `gamma` — Tucker uses the regression slope `Cov(total, V)/Var(V)`; Levine uses
  the congeneric effective-length ratio, which differs for an internal anchor
  (`Var(total)/Cov`) versus an external one (`(Var(total)+Cov)/(Var(V)+Cov)`) —
  then equates linearly. Compute in Rust (`equating::equate_neat_linear`); exposed
  via PyO3 and Python. Validated by the exact reduction to equivalent-groups
  linear equating under equal anchor moments (all four Tucker/Levine ×
  internal/external variants, any `w1`, to `< 1e-9`), a hand-computed check that
  pins the internal-vs-external Levine gamma against an independent oracle, and a
  500-replication Monte-Carlo under a common-regression generative model
  (equated-score interior RMSE 0.39 → 0.19 from `N = 1000` to `4000`, ratio 2.02 ≈
  √4; max bias 0.051 → 0.034). Deferred: Levine true-score equating, Braun-Holland.

- **Kernel equating + log-linear presmoothing** (von Davier, Holland & Thayer,
  2004; Holland & Thayer, 2000). Two enhancements to the equating module.
  `loglinear_smooth(counts, degree)` presmooths a score-frequency distribution by
  Poisson-ML log-linear fitting (on an orthonormal polynomial design over a
  centered/scaled score, Newton with step-halving), preserving the first `degree`
  sample moments exactly while damping sampling noise; it returns AIC/BIC so a
  caller can select the degree, and saturated at `degree = k` it reproduces the
  raw relative frequencies. `equate_observed_scores_kernel` adds a Gaussian-kernel
  continuization (von Davier's `F_h(x) = Σ_j r_j Φ((x − a x_j − (1−a)μ)/(a h))`,
  bandwidth by the penalty method) and optional per-form presmoothing to the
  equipercentile family, behind a single extended entry point whose uniform-kernel
  path reproduces the existing equipercentile bit-for-bit. Compute in Rust
  (`equating::loglinear_smooth` / `equate_eg_ext`); exposed via PyO3 and Python.
  Validated by exact-identity anchors — uniform-kernel equating equals the
  equipercentile to `< 1e-12`; presmoothing preserves the first `T` moments to
  `< 1e-8` and reproduces `rel_freq` when saturated; the Gaussian-kernel
  self-equate is the identity, a large bandwidth drives kernel equating to linear
  to `< 1e-4`, and the continuized density preserves the discrete mean and
  variance — plus a 500-replication Monte-Carlo against the population
  Gaussian-kernel transform (interior RMSE 0.53 → 0.26 from `N = 1000` to `4000`,
  ratio 2.03 ≈ √4; max bias 0.049 → 0.020). Deferred: bivariate presmoothing,
  kernel-NEAT, and analytic standard errors.

- **Observed-score equating** (Kolen & Brennan, 2014). A new
  `mlsirm_core::equating` module and the public `equate_observed_scores` /
  `equate_neat` — the raw-score complement to the IRT scale linking (`irt_link`).
  Equivalent-groups mean, linear, and equipercentile equating (percentile-rank
  matching with the Kolen-Brennan uniform-kernel continuization, equated scores
  kept real-valued), and the common-item non-equivalent-groups (NEAT) design via
  chained equipercentile and frequency-estimation (post-stratification)
  equipercentile. The attainable min/max are computed on relative-frequency
  vectors; the frequency-estimation synthetic densities are renormalized so a
  poorly overlapping anchor degrades toward each group's own marginal rather than
  corrupting the cdf. Compute in Rust; exposed via PyO3 and a Python
  `equating.py` (`EquateResult`). Validated by three exact identities — the
  equipercentile self-equate is the identity to `< 1e-9` (including the low
  boundary at `x = 0`), mean/linear recover a known integer-affine transform to
  `< 1e-9`, and both NEAT methods collapse to EG equipercentile under equal
  anchor distributions to `< 1e-9` — plus a 500-replication Monte-Carlo against a
  deterministic Lord-Wingersky population equating: the empirical equipercentile
  converges at the expected rate (interior RMSE 0.53 at `N = 1000` → 0.26 at
  `N = 4000`, ratio 1.99 ≈ √4; max bias 0.068 → 0.031). Deferred (each a drop-in
  behind the density/table interface): Tucker/Levine linear NEAT, log-linear
  presmoothing, and Gaussian-kernel equating (von Davier et al., 2004).

- **Nonparametric polytomous person fit U3poly** (Emons, 2008; van der Flier,
  1982). `u3_person_fit_polytomous(responses, n_cat)` computes van der Flier's
  `U3` person-fit statistic generalized to ordered polytomous items — a
  *model-free* index: each item-step response function `P(Y_i >= m)` is estimated
  by its sample proportion, turned into a logit weight, and a person's observed
  weighted score is compared to the largest and smallest weighted scores
  attainable at that person's total score (the conditioning group), giving
  `U3 in [0, 1]` (1 = maximally popularity-inconsistent). The attainable min/max
  bounds are computed by exact min-plus / max-plus DP (not the flat "sum of the
  top-k weights" shortcut, which over-counts once an unused category breaks
  within-item monotonicity). `u3_cutoff_polytomous(fit, n_persons)` returns a
  simulated `1 - alpha` critical value by parametric bootstrap (U3poly has no
  usable analytic null; Emons used simulated critical values). Compute in Rust
  (`mlsirm_core::poly::u3_poly_person_fit` + `u3_poly_bootstrap_cutoff`).
  Validated by an exact `n_cat = 2` reduction to a from-scratch van der Flier `U3`
  (max abs diff `< 1e-10`) and a 500-replication Monte-Carlo (GPCM, `K = 5`,
  `n = 600`): the simulated cutoff calibrates the marginal flag rate under a
  matched population (Type I 0.052 normal / 0.054 skew) and detects careless
  responders with power ~1.00; the per-total-score-group flag-rate deviation
  (0.066 normal / 0.083 skew) is reported to make transparent that a single
  pooled cutoff cannot fully condition on the total score. Complements the
  parametric `l_z`/`l_z*` (`person_fit_polytomous`) with a distribution-free
  screen.

- **Polytomous M2 limited-information goodness-of-fit** (Maydeu-Olivares & Joe,
  2014). `m2_polytomous(responses, fit)` returns the test-level M2 statistic,
  `df`, `p_value`, RMSEA2 (with a 90% interval), and SRMSR for a fitted GRM/GPCM
  — the ordered-category generalization of the binary M2 (`m2_stat`). It uses
  the cumulative marginals `P(Y_i>=c)` and `P(Y_i>=c, Y_j>=d)` (the same M2 as
  the paper's category-equality form) and reduces **exactly** to the binary
  `m2_rmsea2` at `n_cat = 2`. Compute in Rust (`mlsirm_core::fitstats::poly_m2`),
  reusing the one-Cholesky residual-projection solve. `df = n(K-1) +
  C(n,2)(K-1)² - nK`. Validated by the exact `K=2` reduction (GRM and GPCM) and
  a 500-replication Monte-Carlo: under a matched `N(0,1)` ability `mean(M2)/df =
  0.99` with Type I error 0.05 (nominal), and under a skewed population `M2`
  inflates 4× with power 1.00.

- **Generalized S-X² item fit for polytomous models** (Kang & Chen, 2008, 2011).
  `item_fit_polytomous(responses, fit)` returns the per-item summed-score
  chi-square, `df`, `p_value`, and retained cell count for a fitted GRM/GPCM,
  extending the binary Orlando-Thissen S-X²: persons are grouped by summed
  score, and the model-expected category proportions come from the generalized
  Lord-Wingersky recursion (Thissen, Pommerich, Billeaud & Williams, 1995) with
  the leave-one-out summed-score distribution. Boundary score groups are merged
  and adjacent categories collapsed to a minimum expected frequency. Compute in
  Rust (`mlsirm_core::poly::poly_s_x2`), exposed via PyO3. Validated to reduce
  **exactly** to the trusted binary `fitstats::s_x2` at `n_cat = 2` (GRM and
  GPCM, statistic and df), and — at the true generating parameters — to track
  its reference chi-square (`E[S-X²] ≈ Σ cells`) for both the GPCM (2008) and
  GRM (2011) families.

- **Marginal (MMLE-EM) estimation for the full latent-space family.**
  `fit(estimator="mmle")` now fits `MIRT`/`MLS2PLM`/`MLSRM` (and `ULS2PLM`/
  `ULSRM` under a population structure) by Bock-Aitkin-style marginal EM:
  person latents `(theta, xi)` are integrated over Gauss-Hermite grids —
  tractable via the simple-structure conditional factorization — with a
  Fisher-preconditioned GEM M-step and the Jeon et al. (2021) LSIRM priors as
  MAP penalties (`PenaltyConfig::lsirm_prior`). Rust core
  (`mlsirm_core::marginal`) with a NumPy mirror
  (`fast_mlsirm.estimators.marginal`) held to 1e-9 end-of-run parity
  (`tests/test_marginal_parity.py`); design and paper basis in
  `docs/mmle_marginal_lsirm_design.md`.
- **Estimation-level multigroup and multilevel population structures** for the
  marginal estimator: `fit(..., group_id=...)` (Bock-Zimowski group trait
  means/SDs, common items, pinned reference group) and
  `fit(..., cluster_id=...)` (Fox-Glas random intercept, `sigma_u`/ICC
  estimated). Results surface on `FitResult.population` and persist through
  `save_fit_result`; the CLI `fit` command gains `--estimator`, `--group-id`,
  `--cluster-id`, `--q-theta`, `--q-xi`, `--q-u`, and `--tolerance`.
- **wgpu E-step kernels for the marginal estimator**
  (`mlsirm_core::gpu_marginal`): the E-step hot path runs in f32 on the GPU
  with the same race-free slot-ownership reduction as the JML kernels, cutting
  a 31k-person multilevel E-step iteration from ~110 s (CPU f64) to ~5 s on a
  laptop RTX 3050 Ti; the M-step and final EAP pass stay on the CPU in f64,
  and hosts without an adapter fall back to the CPU path unchanged.
- **Likelihood-based fit statistics** (`fast_mlsirm.fitstats`): Orlando-Thissen
  S-X² via the Lord-Wingersky recursion generalized to the joint `(theta, xi)`
  grid (chi-square tail without SciPy), Benjamini-Hochberg FDR control,
  Drasgow `l_z` and Snijders `l_z*` person fit with the MAP `r_0` correction,
  and infit/outfit at the marginal EAPs.
- **M2 limited-information goodness-of-fit** (`fast_mlsirm.fitstats.m2`;
  Maydeu-Olivares & Joe 2005/2006, Cai & Hansen 2013): the M2 statistic on the
  univariate + bivariate residual margins, its df and χ² tail p-value, the
  RMSEA2 approximate-fit index with a 90% noncentral-χ² confidence interval,
  and the bivariate SRMSR (Maydeu-Olivares 2013). Every model-implied margin
  (and the up-to-4th-order entries of the multinomial residual covariance
  `Xi_2`) is computed exactly by the local-independence factorization over the
  `(theta, xi)` node set — `pi_S = Σ_c w_c ∏_{i∈S} P_i(c)` — the same
  factorization the E-step already uses (Cai-Hansen); the derivative matrix
  `Delta_2` is central-differenced from the node moments and the quadratic form
  is evaluated through one Cholesky of `Xi_2` (never an explicit inverse). Rust
  core (`mlsirm_core::fitstats::m2_rmsea2`, kind-aware) with a NumPy reference
  held to 1e-6 parity; well-specified-vs-local-dependence calibration tests in
  both suites.
- **GPU EAP scoring kernel** (`mlsirm_core::gpu_marginal::score_eap_gpu`, WGSL
  `score_pass`): Bock-Mislevy (1982) EAP scoring on the wgpu path, one thread
  per person (race-free — each person owns its output slots, unlike the E-step
  reduction), reusing the same `cell_l` binary-sparsity table decomposition.
  Exposed as an **opt-in** device on `score_eap_device(..., Device::Gpu)` and
  through PyO3 `score_bank_eap(..., device=...)` and
  `serving.score_respondents(..., device="gpu")`; the default stays the exact
  f64 CPU reduction, so precision-sensitive paths and serving parity are
  unchanged. f32 kernel, GPU-vs-CPU parity ≤ 2e-3 verified on-device
  (`gpu_eap_matches_cpu_reduction`); falls back to CPU with no adapter or when
  `n_dims`/`latent_dim > 8`. Extends GPU offload from the E-step to the 31k-
  person serving hot path (project compute policy: all math in Rust, GPU where
  it pays).
- **IRT scale linking for common-item designs** (`fast_mlsirm.irt_link`;
  `mlsirm_core::linking`): the moment methods (mean/mean, mean/sigma) and the
  characteristic-curve methods of Haebara (1980) and Stocking & Lord (1983) for
  putting a separately-calibrated new form onto the reference scale
  (`theta_old = A·theta_new + B`), motivated by the mixed-format / multi-study
  linking papers in the corpus (Kim & Lee 2006; Yao & Boughton 2009; Brossman &
  Lee 2013). The characteristic-curve loss is minimized by a self-contained
  Nelder-Mead over `(A, B)` from the mean/sigma start, integrated over a
  standard-normal Gauss-Hermite grid. Rust compute path; recovery tests for all
  four methods in both suites. (Complements the existing anchor-based
  `link_fixed_item_parameters` and the FIPC serving path.)
- **Item screening pipeline** (`fast_mlsirm.select_items`): iterative
  fit → flag → remove → refit with sparse / S-X²-BH / mean-square band /
  low-discrimination / map-isolation flags, an `l_z*` person screen, a
  per-dimension item floor, and a full audit trail.
- **Serving bundle + frozen-parameter scoring** (`fast_mlsirm.serving`):
  schema-versioned JSON bundle of the calibrated item parameters and
  population block, and `score_respondents()` EAP scoring of new response
  payloads with items frozen — the fixed-parameter serving pattern used by
  the downstream importance-assessment API. `fast-mlsirm score` scores a JSON
  payload (or `.npy` matrix) against a bundle from the command line.

- **QMC-EM and MC-EM integration rules** for the marginal estimator
  (`FitConfig(xi_rule="qmc"|"mc", xi_points=..., xi_seed=...)`): the
  latent-space integral runs on Halton low-discrepancy points (randomized-QMC
  shift optional; Jank 2005) or seeded Monte Carlo draws (Wei & Tanner 1990;
  Meng & Schilling 1996) instead of the tensor Gauss-Hermite grid — enabling
  `latent_dim > 3` and better error scaling per node. Both constructions are
  deterministic and bit-mirrored across the Rust/NumPy backends.
- **Rust scoring module** (`mlsirm_core::scoring`, exposed via
  `_core.score_bank_eap` / `score_bank_map` / `eapsum_tables`): EAP
  (Bock & Mislevy 1982), MAP (posterior Newton with observed-information
  SEs), and summed-score EAP conversion tables via the Lord-Wingersky
  recursion (Thissen et al. 1995; Cai 2015), all under per-dimension
  `N(mean_d, sd_d^2)` priors that cover single, multigroup
  (`mu_g, sigma_g`) and multilevel populations (conditional
  `N(u_hat_c, 1)` or marginal `N(0, sqrt(1 + sigma_u^2))`).
  `score_respondents(..., method="eap"|"map"|"eapsum", prior=...)` and the
  bundle's embedded `eapsum_tables` expose these to serving.
- **Fit statistics moved to the Rust core** (`mlsirm_core::fitstats`): S-X²,
  Benjamini-Hochberg, `l_z`/`l_z*`, infit/outfit now compute in Rust
  (`fast_mlsirm.fitstats` delegates; the NumPy bodies remain the parity
  reference/fallback). S-X² gains the `rms_residual` practical-significance
  effect size (Sinharay & Haberman 2014) and `select_items` gates its flag on
  `sx2_min_effect`; the mean-square gate now uses infit only (outfit is
  reported, not gating — it explodes under very low pass rates); the person
  screen threshold is configurable and the Snijders `r_0` correction is
  centered on the population prior mean (cluster intercepts / group means).
- **Fixed Item Parameter Calibration** (`fit(..., anchors=...)`): anchored
  items stay frozen (optionally `tau` too) while new items and a freed
  population mean/SD are estimated — the multiple-cycle prior-update (MWU-MEM
  style) variant Kim (2006) found robust; latent-space orientation inherits
  from the anchors (no PCA re-alignment). **Concurrent calibration** is the
  existing multigroup path with structural missingness (Hanson & Béguin
  2002), covered by a dedicated recovery test.

### Changed

- `estimator="mmle"` with a spatial/multidimensional model now fits (routed to
  the marginal estimator) instead of raising `NotImplementedError`; plain
  `ULS2PLM`/`ULSRM` without a population structure keep the legacy
  unidimensional fast path and its exact previous behavior.

- Exposed the Rust MMLE-EM estimator (`mlsirm_core::mmle::fit_mmle_2pl`) through
  the PyO3 binding as `fast_mlsirm._core.fit_mmle_2pl`, so
  `fit(estimator="mmle")` now runs on the Rust core when the extension is built
  (previously it always fell back to the NumPy reference). To keep the two
  backends statistically equivalent, the Rust core's Gauss-Hermite table was
  aligned from 21 to 41 nodes, bit-identical to the NumPy reference's default
  `hermegauss(41)` quadrature; `tests/test_rust_parity.py` gains MMLE parity
  tests (a/b/theta agreement at the shared EM optimum, measured ~1e-8).

- Made the Rust core (`fast_mlsirm._core`) the **primary** numeric path: the
  default `FitConfig.backend` and CLI `--backend` are now `"auto"`, resolving to
  Rust when the compiled extension is available and falling back to the NumPy
  reference otherwise. The verified LSIRM/MLS2PLM neg-loglik, gradient, and
  distance-kernel formulas are ported bit-for-bit; observable outputs are
  unchanged.

### Added

- GPGPU acceleration of the negative-log-likelihood and gradient hot path inside
  the Rust core via [wgpu](https://github.com/gfx-rs/wgpu) (MIT/Apache-2.0),
  exposed as a device sub-option of the Rust backend rather than a separate
  compute-backend axis. Select with `FitConfig(backend="rust", rust_device=...)`
  or `fast-mlsirm fit --backend rust --rust-device {auto,cpu,gpu}`; the GPU path
  falls back to the identical CPU implementation at runtime when no GPU adapter
  is available. Added requested-device provenance on `FitResult.rust_device`
  and in `fit_summary.json`, plus numerical-parity tests asserting the Rust
  device paths match the NumPy reference.
- Added `docs/papers/README.md` with a citation and canonical link for Wu et al.
  (2021, arXiv:2108.11579), grounding fast, accelerator-friendly IRT estimation
  without vendoring the PDF into the repository.
- Added `tests/test_rust_parity.py`, a Rust<->NumPy numerical parity gate that
  asserts agreement to `1e-6` across all five model variants, multiple problem
  sizes, and masked/dense fixtures (observed difference ~1e-13).
- Added a Rust toolchain plus a resolved-default-backend assertion to the
  `python` CI job so the primary Rust path is built and exercised by the suite.
- Added `scripts/release_acceptance.py` to execute a sales-readiness end-to-end
  smoke: simulate, fit (auto + optional rust), diagnostics, and report rendering.
- Added `docs/release_acceptance.md` to document acceptance inputs, outputs, and
  pass criteria.
- Added `docs/enterprise_sales_readiness.md` and `scripts/sales_readiness.py`
  to produce a machine-readable enterprise procurement readiness manifest.
- Added aFIPC-style fixed-item calibration diagnostics and
  `diagnose-fixed-item-calibration` to select candidate probability tensors
  with kaefa-style item-fit penalty evidence.

### CI

- Replaced package-only Rust smoke with release-acceptance execution in CI.
- Added an enterprise sales-readiness gate to validate acceptance evidence,
  policy documents, package artifacts, installed-version consistency, and Rust
  backend import proof.

### Documentation

- Updated commercial-readiness and README documents to point to the acceptance
  checklist and execution command.
- Added KRW 2,000,000,000 enterprise sales-review criteria and explicit go/no-go
  evidence requirements.
- Updated the Figma product design packet with Information Architecture,
  화면정의서, key screen, wireframe, and user stories for fixed-item
  calibration review.

### Added

#### Rust-only literature true-parameter recovery gate

- A Rust-only true-parameter recovery experiment for a bounded representative
  Kang and Jeon (2025) MLS2PLM simulation cell (`P = 500`), tracing the
  simple-structure equation, sign convention, identification handling,
  recovery metrics, and citations.
- Orientation-invariant latent-map recovery metrics covering item parameters,
  person traits, person and item interaction positions, and distance weights.
- Scheduled, manual-dispatch, and release-tag statistical-study workflows that
  execute exhaustive ignored Rust studies in exact-name-validated shards while
  pull-request CI retains bounded CPU/GPU sentinels.
- A source-backed finite-Monte-Carlo convergence floor
  (`p0 - 2 * sqrt(p0 * (1 - p0) / R)`) for the 500-replication higher-order
  DINA recovery study.

#### Fail-closed Vuong selection summary

- A bounded public `compare_nonnested_models` orchestration API that preserves Rust-computed casewise likelihood-ratio mean, variance scale, corrected selection statistic, and two-sided probability together with explicit model-relation metadata when the normal-selection kernel is applicable.
- Auditable `ModelRelation`, `ComparisonStatus`, and immutable `ModelComparisonResult` contracts.
- Relation-appropriate routing for nested, boundary-nested, overlapping, strictly non-nested, and unknown candidate pairs.

#### Rubric blueprint compiler

- Versioned rubric and rubric-level schemas with explicit construct, observable evidence, task-family, response-format, locale, and prohibited-pattern contracts.
- Deterministic bounded compilation across task family, difficulty band, evidence mode, and replicate cells.
- Full SHA-256 rubric, blueprint, and generation-contract fingerprints plus authoritative 128-bit public blueprint and contract handles; 64-bit convenience display identifiers remain explicitly non-authoritative, and 64-bit digest slices also seed deterministic generation.
- A prompt-injection boundary and strict generated-item JSON Schema 2020-12 contract without adding a hosted-model SDK or network dependency.
- Immutable rubric and blueprint provenance constants in generated-item schemas, preventing wrong-blueprint replay from passing structural validation.
- Response-format-specific, closed, bounded answer-key contracts and ordered score-level schemas that require every rubric score exactly once.
- Explicit text and collection bounds for model-generated content and provenance fields.
- A deterministic standard-library changelog-fragment renderer; files in `docs/changelog.d` are authoritative `Unreleased` release notes and are validated as part of the repository test suite.
- Evidence-Centered Design documentation and a production roadmap from provider adapters through Rust-backed calibration and governed item-bank lifecycle.

#### Rust bifactor scoreability indices

- Rust-native continuous-indicator bifactor scoreability diagnostics: ECV-SS,
  ECV-SG, ECV-GS, item ECV, strict-pattern PUC, omega total, omega
  hierarchical, and construct replicability H.
- An explicitly named logistic latent-response conversion for fitted orthogonal
  bifactor slopes. Its omega values are documented as continuous
  latent-response coefficients, not categorical observed-score reliability.
- A modular PyO3 `_bifactor_core` surface and immutable typed Python API:
  `bifactor_scoreability`, `bifactor_scoreability_from_logit_slopes`, and
  `BifactorScoreabilityResult`. Python validates shapes and marshals results;
  all scoreability arithmetic remains in Rust.
- Fail-closed structural validation requiring every item to load on the
  declared general factor, uniquenesses in `[0, 1]`, and the standardized
  identity `sum(lambda^2) + uniqueness = 1` within `1e-8`.
- Formula-oracle, Rust/Python parity, structural, numerical-stability,
  logistic-conversion, and package-export tests plus buyer-facing
  interpretation boundaries.

#### Governed rubric item generation

- Bounded source-document packets with exact-content SHA-256 provenance and redacted audit metadata.
- Content-addressed generation requests that bind one rubric contract, blueprint, seed, and evidence-mode-valid source packet.
- A runtime-checkable provider protocol and deterministic offline fixture provider without hosted SDK, credential, or network dependencies.
- Strict provider-JSON decoding that rejects duplicate keys, non-finite numbers, oversized output, excessive nesting depth, missing fields, and unknown fields.
- Immutable rubric and blueprint replay protection across ids, 128-bit audit handles, full fingerprints, and governed rubric versions.
- Exact ordered rubric-score coverage, response-format-specific typed answer keys, option/key consistency, source-id resolution, and verbatim evidence-span validation.
- Explicit pairwise left/right/tie semantics with null-only tie preferences.
- Deterministic request, candidate, and execution fingerprints plus provider-failure redaction that omits raw source and generated text.
- Public generation, candidate, answer-key, attribution, and execution APIs with complete package exports.
Structural validation remains separate from semantic review, psychometric calibration, DIF, local-dependence, exposure, drift, and governed item-bank acceptance.

#### Adaptive factor rotation and criterion selection

- Rust-native adaptive exploratory factor rotation with a broad criterion
  registry, orthogonal and oblique gradient-projection optimization,
  deterministic multi-start search, coarse CPU multithreading, and explicit
  convergence/basin diagnostics.
- Criterion-neutral empirical selection using stability, simple structure,
  degeneracy, target recovery, bootstrap Tucker congruence, Pareto evidence,
  and declared decision policies. Objective values are never compared directly
  across criterion families or described as a proven global optimum.
- Modular PyO3 `_rotation_core` bindings and package-root Python APIs for
  criterion discovery, analytic value/gradient evaluation, multi-start
  rotation, and typed immutable solutions.
- Symmetric positive-definite Cholesky log-determinant/inverse handling for the
  Bentler criterion, including pivot-provoking and near-singular regression
  oracles.
- GPArotation-compatible complete/partial target semantics using binary
  zero-or-one masks and the loss `sum(w * residual^2)`. Continuous weights are
  available only through the separately named `lp_wls` kernel.
- An explicit scope boundary: Promax, Cubimax, iterative Lp/FSS orchestration,
  cluster/EIV/echelon procedures, user-defined compiled criteria, and a
  parity-verified wgpu batch optimizer are not part of this release slice.

#### Hourly pull-request governance

- A read-only hourly GitHub Actions loop that runs the existing pull-request queue governance evidence builder, publishes its JSON and accessible HTML audit artifacts, and retains native branch-protection and auto-merge gates instead of bypassing review or required checks.

### Changed

#### Rust-only literature true-parameter recovery gate

- The duplicate NumPy-only recovery experiment is removed; the Rust core is
  the single evidence path for literature recovery gates.
- The historical `cdm::tests::mc_ho_recovery_500` study is removed at the
  source level. Its generating design, fixed seeds, and RMSE, bias, and
  agreement thresholds are preserved verbatim by the reviewed
  `higher_order_dina_recovery_respects_monte_carlo_tolerance` integration
  study, which gates convergence on the documented two-standard-error binomial
  floor instead of an exact finite-sample proportion.

#### CI queue and review-governance hardening

- Pull-request CI runs now share a PR-number-scoped concurrency group, so a newer head cancels superseded queued or running CI evidence instead of consuming capacity for an obsolete commit.
- Push CI remains isolated by branch or ref and does not collide with pull-request validation.
- Draft pull requests no longer consume automatic CodeRabbit reviews, and automatic incremental review-on-every-push is disabled; maintainers request a final review only after a stable head is ready.
- The hourly read-only PR-governance workflow verifies its repository contract with the Python standard library, fails closed when no matching test is discovered, and no longer assumes that `pytest` is preinstalled on a fresh scheduled runner.
No test, security, packaging, coverage, or merge requirement is weakened by these operational changes.

### Fixed

#### Rust-only literature true-parameter recovery gate

- Ignored-test shard discovery rejects stale skip declarations, duplicate
  skips, ambiguous final-component exclusions, and silently empty shards.
- Explicit-GPU parity evidence fails closed when the Vulkan adapter is
  unavailable instead of silently skipping.

### Security

#### Fail-closed Vuong selection summary

- Omitted relation metadata defaults to `unknown`.
- Nested, boundary-nested, and unknown relations are routed before the non-nested normal-selection kernel is invoked, so a rejected or exact-zero non-applicable statistic cannot mask the required likelihood-ratio or relation-resolution procedure.
- The API does not report a winning model until Vuong's formal first-stage distinguishability evidence is available from a common compiled score/information contract.
- Numerical variance checks are not mislabeled as the formal weighted-chi-square distinguishability test.
- Casewise inputs are bounded and normalized to finite floats before FFI; booleans, opaque values, non-finite values, conversion overflow, malformed labels, invalid parameter counts, and compiled-kernel rejections fail closed without leaking low-level exception text or reproducing statistical arithmetic in Python.

## [0.1.2] - 2026-07-31

### Added
- Full paired-comparison / rating / inter-rater stack on main (PR #374 integrating the #290–#328 seonghobae chain tip): Thurstone Case V, Bradley–Terry MM, LSR/I-LSR, Rank Centrality, Plackett–Luce rankings and top-1, Kendall circular triads / *u*, Elo / Glicko / Glicko-2 / Stephenson / multiplayer Elo / FIDE, prediction metrics, BRATT ties model, Fleiss/Light kappa, ICC, Krippendorff α, Finn, Maxwell RE, Robinson *A*, mean pairwise Pearson/Spearman, Stuart–Maxwell / Bhapkar marginal homogeneity, rater bias, and Cohen kappa sample-size helpers — Rust core + PyO3/Python API + unit/paper tests.
- DeepWiki badge on the primary README docs surface (PR #373).

### Changed
- Stack features land without regressing the simple-structure MLS2PLM NLL path, coarse person-shard multithreading, or PRIMARY-only wgpu GPU init with soft f64 CPU fallback (preserved from v0.1.1 / PRs #371–#372).

## [0.1.1] - 2026-07-31

### Fixed
- Coarse fixed-shard Rust multithreading for the JML `neg_loglik_and_grad` hot path (`thread::scope` person shards, N≥256) with bit-identical reduction vs single-thread (PR #371).
- wgpu GPU init uses `Backends::PRIMARY` only (no GL/EGL), so sandboxes with broken `/dev/dri` soft-fail to the f64 CPU path instead of SIGSEGV (PR #371).
- Paper-grounded simple-structure MLS2PLM formula contract reaffirmed (Kang & Jeon 2025; Jeon et al. 2021); no formula drift.

### Changed
- Unit test forces multi-worker NLL shards and compares all gradient blocks to the single-thread reference.

## 0.1.0 - 2026-07-02

### Added

- MLS2PLM simulation, fitting, diagnostics, and HTML report generation.
- Optional Rust/PyO3 backend exposed as `fast_mlsirm._core`.
- Backend selection through `FitConfig.backend` and `fast-mlsirm fit --backend`.
- Fit summary persistence of the resolved backend.
- Commercial beta readiness documentation, support policy, security policy, and
  release verification checklist.

### Known Limits

- Current estimators are regularized point-estimate JML/MAP-style workflows,
  not Bayesian posterior samplers.
- Ordinal response estimators, sparse/block execution, benchmark automation,
  and posterior predictive checks remain future work.
