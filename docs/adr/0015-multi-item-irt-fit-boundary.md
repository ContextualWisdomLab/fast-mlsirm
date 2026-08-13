# ADR-0015: Multi-item IRT fit boundary and experiment readiness

Status: **Proposed**
Date: 2026-08-13
Supersedes: none
Superseded by: none

## Context

An IRT result is an item response matrix, not a scalar judge score. The
cross-component contract already requires at least two dichotomous or two
polytomous item columns, but several public numerical fitters still accepted a
one-column matrix before delegating to Rust. That let a caller bypass the
integration contract by calling a fitter directly. Low-level numerical and
diagnostic primitives may still be useful with one item; an interpretable IRT
experiment is not.

LLM-judge outputs add a second risk. A matrix can have the right shape while
having too few persons, an item with almost no observations, a constant item,
or insufficient factor anchors. Such a fit must not be reported as stable
evidence. These checks are measurement gates, not keyword or lexical
judgments, and all LLM-as-a-Judge calls remain routed through
`contextual-orchestrator`. The readiness helper was initially only exercised
by unit tests, so the CLI/benchmark path could still reach native fitting
without this gate.

## Decision

1. The shared `validate_irt_response_matrix` contract is authoritative for
   cross-component data and public IRT model fitters. It requires a 2-D
   persons-by-items matrix with at least two item columns, finite integer
   observations, binary values for dichotomous items, and an explicit bounded
   category count for polytomous items.
2. Every public binary and polytomous estimator entry point calls the shared
   response-matrix validator after its family-specific missing-value
   normalization and before native computation. This includes MLSIRM, 2PL,
   MH-RM, GRM, GPCM, nominal, RSM, unidimensional polytomous, latent-space
   polytomous, and nominal-polytomous entry points.
3. `fit_irt_experiment` is the production/benchmark boundary. It calls
   `validate_irt_experiment_readiness`, accepts scalar factor IDs or per-item
   factor memberships for anchor coverage, and invokes the selected numerical
   fitter only after the gate passes. The CLI `fit` command uses this boundary;
   the CLI `diagnose-dimensions` command validates its source matrix and every
   cross-validation training fold through the same boundary. Direct numerical
   fitters and the default low-level diagnostic mode remain available for
   explicitly diagnostic work and must not be presented as stable experiment
   evidence.
4. `validate_irt_experiment_readiness` is the pre-interpretation gate. Its
   defaults require at least five persons, three observed responses per item,
   two observed values per item, and, for polytomous inputs, at least one
   observed response in every declared category for every item. When factor
   labels are supplied, it also requires two items per factor anchor.
   Controls and factor containers are validated before counting; strings,
   booleans, malformed arrays, and unhashable labels fail closed.
   The shared item-type and category-count controls accept only exact built-in
   strings/integers before set membership or range comparisons.
   These exact numeric defaults are package-owned conservative evidence gates,
   not universal sample-size or identification guarantees. Samejima (1969)
   and Muraki (1992) motivate retaining observed variation and declared
   category support for graded/GPCM item parameters; Cai (2010b) and Reckase
   (2009) motivate explicit factor anchors for confirmatory multidimensional
   identification; Jones and Loe (2013) and Iannario et al. (2022) motivate
   treating category-count choices as calibration questions rather than
   assuming that more categories improve measurement. No cited primary study
   mandates the exact `5`, `3`, `2`, or `2` cutoffs, so they must be reported as
   conservative package controls and revalidated with simulation/gold data.
5. A failed shape or readiness gate is retained as a failed comparison and
   cannot be repaired by keyword matching, positional category repair, silent
   item dropping, or a blind retry. `LLMJudgeResult.to_irt_row` continues to
   require multiple criteria.
6. `category_method="binary_threshold"` is the default for an explicit
   polytomous `category_count` when callers omit `category_method`. It asks one
   strict Boolean question per ordered boundary and criterion, derives
   categories only from validated monotone thresholds, and fails the complete
   comparison on malformed or non-monotone output. The method is bounded at 64
   provider calls per result and must record its call count, latency, usage,
   parse status, and category row. Equal-width K-way `direct` output remains
   available only when explicitly requested for calibration; cumulative
   thresholds are also explicit. When the injected contextual-orchestrator
   exposes its already-bounded `client.local_concurrency`, independent
   boundary calls may run concurrently through that limit; otherwise the
   adapter remains sequential. Output order is deterministic and monotonicity
   is validated only after all returned boundaries are assembled. A malformed,
   unavailable, or non-monotone boundary raises `JudgeFormatError` with bounded
   `.evidence` containing ordered boundary records, call/parse status, partial
   trace-step counts, and aggregated usage; callers must retain that evidence
   as a failed calibration comparison rather than discarding the exception.
   Callers may provide exactly K `category_anchors` on every criterion. When
   present, each requested boundary receives its matching anchor as rubric data;
   mixed, incomplete, or mismatched anchor sets fail before provider calls.
   Omitted anchors remain an exploratory compatibility mode and are not gold
   calibration evidence.

## Implementation Plan

- `python/fast_mlsirm/llm_judge.py`: keep direct and cumulative methods
  compatible, make the bounded binary-threshold method the safe implicit
  polytomous default, and derive only from strict Boolean boundary responses.
  Validate complete per-criterion `category_anchors` when supplied and bind the
  current anchor to each Boolean boundary without putting rubric text in the
  system message.
  Reuse an exposed contextual-orchestrator local-concurrency bound for
  independent boundary calls without adding a provider client or a fallback
  transport.
- `tests/test_llm_judge.py`: cover call decomposition, weighted category/IRT
  projection, anchor validation/binding, malformed/non-monotone responses,
  bounded gateway concurrency, failure evidence retention, and the 64-call
  resource cap.
- `README.md` and this ADR: document the method as calibration-only and retain
  the no-keyword/no-positional-repair boundary.
- `contextual-orchestrator/docs/benchmarks/` and ADR 0006/0008: record paired
  MLX evidence with parse failures, calls, latency, usage, and semantic limits.
- Verification: run the focused judge/IRT tests, the complete Python suite with
  the current PyO3 extension materialized, and a live MLX smoke through the
  contextual-orchestrator route.

## Invariants and acceptance evidence

- `tests/test_irt_contract.py` covers the shared matrix domain, readiness
  controls, factor-membership boundaries, representative public fitter
  rejection of one-item input, and native-call prevention at
  `fit_irt_experiment`.
- Every public estimator entry point listed in the decision validates the
  multi-item response shape before native-core loading; the production-boundary
  missing-mask regression proves that zero-filled missing cells cannot satisfy
  the observation minimum.
- Production dimension diagnostics reject an unready source matrix before fold
  construction and reject an unready training fold before invoking `fit`.
- The targeted IRT/judge/security suite passes with the CLI production path
  behind `fit_irt_experiment`.
- Each benchmark records item count, category count, person count, observed
  count per item, factor coverage, parse status, provider readiness, and model
  trace metadata before reporting fit or bias.
- A polytomous K-sweep uses explicit K values and preserves category occupancy;
  it does not infer a positive bias from a single K or from keyword matches.
- Binary-threshold calibration never uses keywords, category positions, or
  silent output repair; an unparseable boundary invalidates that comparison.
  Concurrent boundary calls are still bounded by contextual-orchestrator and
  retain request order in the success or failure evidence record. A failed
  boundary record is bounded and contains no source answer copy beyond a
  short output preview.
- If category anchors are supplied, every criterion has exactly K non-empty
  definitions and the boundary prompt carries only the matching anchor as
  untrusted rubric data; anchor presence is recorded in the result. Missing
  anchors are explicit exploratory evidence, never silently inferred labels.

## Consequences and trade-offs

- Direct callers with a one-item matrix receive an early, package-owned
  `ValueError` instead of a native fit or an uninterpretable estimate.
- Small pilot data can still be inspected through numerical/diagnostic APIs,
  but the CLI and benchmark boundary rejects it unless it meets the readiness
  defaults; no silent override is provided.
- The shared validator is called after each model's existing missing-value
  normalization. This preserves the package's established NaN/negative
  missing conventions without duplicating native numerical logic.

## Alternatives considered

### Enforce the rule only in the LLM judge

Rejected: direct numerical callers could still bypass the contract, and the
same measurement boundary would behave differently depending on its producer.

### Reject one-item input in every low-level Rust kernel

Rejected: item-level diagnostics and numerical unit tests have legitimate
single-item uses. The experiment boundary, not every primitive, is the right
scope.

### Convert a scalar into a synthetic second item

Rejected: this creates pseudo-replication and invalid information for IRT.

### Use keyword or category-position matching after an LLM parse failure

Rejected: it is not a measurement model, is vulnerable to prompt/output
format changes, and violates the fail-closed judge contract.

## Failure, migration, and follow-up register

| Finding | Direction | State |
| --- | --- | --- |
| Older callers may pass one item to a public fitter | Build a real multi-item matrix or use an explicitly diagnostic API; do not pad or duplicate a column | Implemented on this branch |
| Shape validation could pass a matrix too small for stable interpretation because the readiness helper had no production call site | Add `fit_irt_experiment` as the production/benchmark gate, preserve missing cells as NaN, accept factor memberships for confirmatory coverage, and keep the numerical fitter behind the gate | Implemented on current head; exact-head review follow-up required |
| `diagnose-dimensions` could bypass the production gate inside its cross-validation loop | Validate the source matrix before fold construction and pass every NaN-preserving training fold through `fit_irt_experiment`; keep the default low-level diagnostic mode explicit | Implemented on current head; exact-head review follow-up required |
| A declared polytomous category can be absent or severely imbalanced | Require every declared category to be observed at least once per item at the production/benchmark readiness boundary; retain detailed frequency imbalance for calibration reports and keep low-level diagnostic fitters available | Implemented on current head; exact-head review follow-up required |
| Runtime subclasses or unhashable values could reach IRT item-type/category-count checks before package-owned validation | Reject non-built-in item types and category counts before membership or range comparisons, with regression coverage | Implemented on current head; exact-head review follow-up required |
| Factor labels may not provide two anchors per dimension | Pass scalar factor IDs or per-item factor memberships through `fit_irt_experiment` and require an explicit design exception for exploratory diagnostics | Implemented on current head; exact-head review follow-up required |
| A fresh 2026-08-13 contextual-orchestrator/MLX K sweep produced valid two-item rows but changed criterion categories as K increased (`(1,0)`, `(2,0)`, `(4,0)`, `(6,2)` for K `2,3,5,7`) from one person | Retain the complete paired observation as calibration evidence, require multiple persons and declared-category occupancy before readiness/IRT interpretation, and do not infer a positive-bias law from the one-person shape smoke | Observed; calibration remains required |
| A fresh two-case 3B MLX probe through contextual-orchestrator produced direct scores of `0.5 -> 1.0 -> 1.0` for the safe case and `0.0 -> 0.0 -> 0.3333` for the unsafe case at K `2,5,7`; cumulative thresholds parsed only at K=5 and failed JSON/monotonicity at K=2/7 | Retain every result and failure in the calibration denominator; do not promote direct or cumulative to an unbiased default. Add an opt-in bounded binary-threshold decomposition, compare its extra calls/tokens against paired human/gold anchors, and keep production IRT claims blocked until replicated. | Observed 2026-08-14; binary-threshold method implemented as experimental, calibration remains required |
| The binary-threshold follow-up made each boundary a Boolean call, but the safe case still failed monotonicity at K=5/7 while the unsafe case parsed at score `0.0` with 8/12 calls and `2,606/3,940` tokens | Keep the method fail-closed and experimental; treat its higher call/latency budget and semantic under-recognition as measured trade-offs, not as bias removal. Do not short-circuit or synthesize higher categories without an explicit ordinal measurement design and held-out gold evidence. | Observed 2026-08-14; follow-up calibration required |
| A fresh same-route direct K-way probe returned the unsafe case at scores `0.0`, `0.5`, and `0.8333` for K=`2,5,7`, while a partial-evidence case returned `0.0`, `1.0`, and `0.0` | Do not let callers silently select among more score identifiers for production polytomous rows. Resolve an omitted method to bounded binary thresholds when `category_count` is present; keep direct K-way output explicit and calibration-only, and retain binary semantic false negatives as failed calibration observations | Implemented in current follow-up; exact-head review required |
| The same binary-threshold probe was stable at score `0.0` for both safe and unsafe cases at K=`5,7`, but under-recognized the safe answer | Treat the safer default as fail-closed measurement protection, not proof of judge quality. Require held-out human/gold recall, category occupancy, and provider/parse denominators before accepting a model or prompt for IRT production | Observed 2026-08-14; calibration gate remains open |
| After this default hardening, PR #816 is exact head `608cfbd39983f485cebe76518c80375e7ff636dd`; all non-skipped required checks are queued, the repository runner API reports `0 total / 0 online / 0 busy`, and no independent current-head approval exists | Keep the PR ready for review but unmergeable. Preserve the exact head and queued/no-runner evidence, request a fresh authorized review, and require terminal exact-head checks plus structured same-head Strix evidence before normal protected merge; local full-suite success cannot substitute for these gates | Goal/ADR expanded 2026-08-14; active infrastructure/review follow-up |
| The real integrated adapter smoke at K=5 with no explicit method used fast-mlsirm `9d18f53` and contextual-orchestrator `a0a354a`: the unsafe answer returned a valid rejected `(0,0)` result after 8 calls, but the safe answer failed closed on non-monotone thresholds after 8 calls | Preserve the default-selection and contextual-routing contract while retaining the safe failure as calibration evidence; never coerce, keyword-match, or retry blindly, and require held-out semantic gold/recall before treating the method as IRT-ready | Observed 2026-08-14; exact integration verified, semantic calibration remains open |
| Binary-threshold failures raised only an exception, so callers could not reliably retain per-boundary parse status, completed-call count, partial trace metadata, or usage for malformed/non-monotone comparisons | Attach bounded `.evidence` to `JudgeFormatError` for binary failures, preserving deterministic boundary order, call/parse status, trace-step counts, usage, and a short output preview; keep source answers out of the evidence and do not retry or repair | Implemented in current follow-up; focused regression required |
| A 3B MLX binary-threshold sweep produced a safe semantic false negative in 6/6 repeated runs at K=5, while generic criteria omitted definitions for intermediate ordinal levels; one anchored probe improved some boundary decisions but still exposed non-monotone outputs | Add optional complete per-criterion `category_anchors`, bind each requested boundary to its explicit rubric definition, record anchor presence, and keep omitted-anchor runs exploratory. Do not coerce non-monotone results, infer missing labels, or claim IRT validity without held-out gold recall | Observed 2026-08-14; anchor contract implemented, live semantic calibration remains required |
| On the same anchored K=5 integrated MLX case, Gemma 4 e4b returned strict `(4,4)` with score `1.0` in `3,031` tokens/`11.96 s`, Llama 3B remained semantically unstable, and Llama 1B failed structured JSON on all eight boundaries | Keep model selection quality/latency evidence based: use the Gemma 4B result only as a candidate, retain 3B false negatives and 1B parse failures in the denominator, and require balanced held-out gold recall before changing a verifier priority or declaring IRT readiness | Observed 2026-08-14; model comparison is exploratory and routed through contextual-orchestrator |
| The current exact-head anchored Gemma 4 e4b rerun completed and parsed all eight Boolean boundaries, but returned evidence-quality `false,false,true,true` and risk-signal `false,true,true,true`; the ordinal gate therefore rejected the complete comparison as non-monotone despite anchors and `3,625` provider tokens. | Preserve the complete semantic failure with trace/usage evidence; do not promote the model or repair threshold order lexically, positionally, by retry, or by coercion. Require balanced held-out gold recall, perturbation stability, and category occupancy before changing verifier selection or treating the result as an IRT observation | Observed 2026-08-14; current exact-head integrated calibration remains required |
| The MLX process was alive enough for `/health` and `/v1/models` HTTP 200 responses while all chat completions timed out with no bytes and accumulated closed/CLOSE_WAIT connections; the first eight judge boundaries consequently failed at the provider boundary with zero completed calls and zero usage. | Separate readiness from liveness, preserve every failed boundary record, use bounded prompt/decode/request concurrency, and restart only the diagnosed single-port inference process through the local supervisor. Do not classify the incident as LibreSSL certificate failure or disable TLS verification. | Observed 2026-08-14; explicit Gemma 4 e4b restart restored direct HTTP 200, lifecycle hardening remains required |
| The real MLX integrated safe-case failure previously exposed only the text `criterion thresholds must be monotone`; the new bounded evidence capture shows all 8 boundaries completed and parsed, 8 trace steps, `2,639` provider tokens, and `semantic_status=non_monotone` | Preserve this structured failure record in the calibration denominator and distinguish semantic non-monotonicity from provider/parse failure; do not turn a complete but invalid comparison into an IRT row | Observed 2026-08-14; live evidence capture verified |
| Binary-threshold boundaries were independent but were previously issued sequentially, making K=7 two-criterion calibration a long serial path | Reuse the injected contextual-orchestrator `client.local_concurrency` as a bounded executor limit; preserve deterministic request order, aggregate trace/usage, and fail closed after complete validation. Keep generic injected orchestrators sequential. | Implemented on current exact follow-up; targeted `58 passed`, full `3630 passed` after native extension build; live K=5/K=7 MLX probe retained with parse/failure and latency evidence |
| The actual contextual-orchestrator `_FastMLSIJudgeAdapter` did not expose its existing gateway client, so the bounded binary executor was discoverable in direct injection tests but not on the integrated judge path. | Keep the fast judge transport-neutral and consume the adapter's exposed `client.local_concurrency` capability when present; retain sequential behavior for generic injected transports and add an integrated adapter smoke before claiming the optimization is active. | Fixed in contextual-orchestrator `d82e592`; exact-source integration smoke reached peak concurrency `2` across `4` boundary calls; linked PR review/check follow-up required |
| A non-monotone binary-threshold failure retained parse status, call counts, trace metadata, and usage but omitted the parsed Boolean boundary values from its bounded evidence, making the semantic failure harder to audit without reopening model output | Record only the validated `meets_threshold` Boolean beside each ordered boundary; keep full responses out of failure evidence and retain fail-closed semantics with no repair or retry | Implemented in current working tree; regression asserts the ordered `false,true` evidence |
| Gemma 4 e4b's anchored binary prompts sometimes treated a minimum boundary as an exact category, rejected stronger answers at lower thresholds, or rewarded merely related/missing-control language; three repeated six-case K=5 runs through the real contextual-orchestrator route produced respectively `4/6`, `5/6`, and `4/6` complete comparisons, with cell accuracy `25%`, `40%`, and `37.5%` among complete rows and repeated over-scoring of partial/unsupported evidence. | Make the binary contract explicit: a boundary asks for "at least" the requested category, stronger evidence remains true, and relevance plus concrete criterion evidence are required; retain non-monotone comparisons fail-closed, preserve all provider tokens/failures in the denominator, and do not call the prompt change bias removal or promote Gemma without larger held-out human/gold, perturbation, and occupancy evidence. | Prompt hardening implemented on current working tree; targeted `62 passed`; live calibration remains exploratory and exact-head review/check follow-up required |
| The remote branch ref advanced to fast-mlsirm `17e19ec90643a8dfcc464cd7dde0b63949539a32`, but GitHub PR #816's `head.sha` and `refs/pull/816/head` still reported predecessor `7605c15456750810e54594b0e2f1ade5ebda6c7d`, and the new commit had no check-runs while predecessor runs remained active. | Treat the remote branch and PR aggregate as separate evidence until GitHub reconciles them; do not force-push, cancel live predecessor checks, self-approve, or interpret the new commit as PR-reviewed. Use a normal subsequent push/re-fetch and request exact-head review only after the PR API, pull ref, required checks, and branch ref agree. | Observed 2026-08-14; active GitHub synchronization/review gate |
| K/order/framing perturbations can change judge scores | Use randomized paired perturbations, human/gold anchors, category occupancy, and parse/provider denominators; never infer a universal positive-K law | Required next |
| A live cumulative-threshold call through contextual-orchestrator parsed into the valid two-item row `(4,0)` but assigned `risk_awareness=0` despite explicit rollback-rehearsal evidence in the answer, while `evidence_quality=4` | Treat valid schema and semantic item accuracy as separate outcomes; retain the miss with trace/usage metadata, add item-level human/gold anchors and balanced held-out recall checks, and do not keyword-match or coerce a replacement category | Observed 2026-08-13; semantic calibration remains required |
| Fast PR #816 was moved from Draft to Ready, then updated with base `origin/main` commit `3d1eab5` in merge head `b3bbeb0431453b07df948714aa08415b031f88af`; the exact merge head passed the full suite (`3630 passed, 2 warnings`) after supplying the current compiled PyO3 extension in the detached verification tree | Keep the base-update evidence separate from remote CI: preserve the native-extension build/preflight contract, rerun exact-head checks after the docs follow-up, and do not treat local full-suite success as independent approval or Strix evidence | Observed 2026-08-13; remote checks and independent review remain required |
| PR #816 later fell behind its protected base after `main` advanced to `fa23aaa`, so predecessor review/check evidence was stale before the judge change could be considered mergeable | Merge the current protected `main` into the PR in an isolated worktree, preserve user-owned files, then rerun the judge/IRT and complete native suites before requesting a fresh exact-head review; never merge a behind branch or reuse predecessor evidence | Remediated in merge head `bb1a2851a9de1fca29f62388391fd3ce7f463f16`; targeted Python `101 passed`, full Python `3654 passed, 2 warnings`, and Rust `994 passed, 118 ignored`; remote approval/check/Strix gates remain required |
| A small local judge model returned a near-schema direct response with `rationale` as an object and criterion scores nested under it, so strict parsing rejected an otherwise usable judgment | Include a complete direct JSON schema example with criterion IDs and explicitly type `rationale` as a string; keep exact-schema parsing and fail-closed behavior | Implemented on current local head; exact-head review follow-up required |
| The same local model then returned all required fields but still used an object for `rationale` | State the non-object rationale boundary explicitly and keep criterion explanations out of that field; do not coerce the object or keyword-match a replacement | Implemented on current local head; exact-head review follow-up required |
| A full local suite initially loaded an older ignored PyO3 binary after the Rust source had gained `m2_cmle_rasch_stat` and `_multilevel_core`, producing misleading missing-export/import failures | Rebuild the repository-root extension with the current `maturin develop --release` configuration before full-suite evidence; verify the required native exports in the same interpreter, and treat stale-ABI failures as invalid test evidence rather than changing numerical code | Observed and corrected locally on 2026-08-13; clean-build CI evidence remains required |
| Current Zotero 9.0.6 Local API is read-only | Revalidate records through the local API; perform file writes only after an authorized Zotero 10+ local-write path or authorized Web API is available | Required follow-up |
| Jones and Loe (2013) is gold OA/CC BY, but SAGE and the author-uploaded ResearchGate PDF are anti-bot blocked from this environment and Zotero item `CWY355RP` has no child attachment | Keep the citation and OA licence evidence, do not reconstruct or import an unauthorised mirror as the original PDF, and retry through an authorized Zotero write-capable route before adding the binary to `docs/papers/` | Observed 2026-08-13; citation-only until an authorized binary is obtained |
| The fast-mlsirm default-branch ruleset allowed zero approvals, stale reviews to survive a push, and unresolved review threads | Require one independent approval, dismiss stale approvals on every push, require approval after the last push, and require resolved review threads while preserving the existing merge methods and no-force-push rules | Protected ruleset updated on 2026-08-13; exact-head CI/review evidence remains required |
| Local `cargo fmt --all -- --check` reports Rust formatting drift already present on `origin/main`, while this branch's implementation diff contains no Rust files and no required CI format context exists | Keep this IRT PR scoped; open a separate formatting-only maintenance PR if the repository adopts a format gate, and do not conceal the baseline drift in a quality report | Observed 2026-08-13; separate cleanup direction recorded |
| fast-mlsirm PR #816 Strix run `31695131004` completed with a green job, but its uploaded artifact contained three `run.json` files with `status: failed`, no head/commit metadata or structured report, and provider failures including NVIDIA NIM `429`, GitHub Models `410` brownout, and context-window exhaustion | Treat the green job as inconclusive pre-fix evidence, never as a clean security scan; keep the PR blocked until the trusted central workflow at its current head rejects metadata-less/failed reports and a later exact-head run produces a validated structured binding after provider recovery | Observed 2026-08-13; central fail-closed fix is on the linked PR and rescan remains required |
| The linked central PR #965 exact head `b8695c534cf15a2227d92f942dcce3c653276393` produced Strix run `31696985802`/job `94436969831` with a green job but no provenance-validation step or `evidence-binding.json`; one completed report lacked head metadata, three reports failed, and provider logs contained NVIDIA NIM `429`, GitHub Models `410`, fail-closed, and no-report markers | Treat the status as inconclusive trusted-base evidence from `pull_request_target`, not as a clean scan for fast-mlsirm. Require the central workflow to be integrated and then emit exact-head structured evidence from a default-branch dispatch before this PR can pass the security gate | Goal expanded 2026-08-13; central doctoring/ADR update and exact-head rescan remain required |
| fast-mlsirm PR #816 exact head `fdbf62dba3beb7fb06768214b241fe568b6b1f48` produced Strix run `31704826853`/job `94462628877` with a green job, but the artifact had one failed provider attempt (`429`), a fallback zero-finding report with no `head_sha`/`commit_sha`, no `evidence-binding.json`, and no provenance-validation step because `pull_request_target` used the trusted base workflow | Treat the green status and fallback narrative as inconclusive base-workflow evidence, not a clean scan. Keep the PR blocked until central fail-closed/provenance changes are integrated and a post-integration exact-head run emits a matching structured binding with clean provider evidence | Goal/ADR expanded 2026-08-13; central remediation `e1cfbed8` pushed, post-merge rescan and independent approval remain required |
| The production readiness boundary validated raw responses before applying the public fitters' `-1`/negative/non-finite and optional mask missing semantics | Normalize the same missing conventions to NaN before readiness counting, retain the callable's mask for the actual fit, and add regressions proving masked and `-1`-coded cells cannot be counted as observations; the CLI inherits the corrected boundary | Implemented on current head; exact-head review follow-up required |
| MH-RM's Python GPCM wrapper accepted `n_cat > 64` although the Rust core contract is `2..=64`, and its docs/error text described categories through `n_cat` instead of `n_cat-1` | Enforce the shared upper bound before native loading, correct the category range documentation and error, and cover the rejected upper boundary | Implemented on current head; exact-head review follow-up required |
| IRT item-variation errors hardcoded two categories and called an item constant even when a caller configured another distinct-value threshold | Report the configured threshold and observed distinct count, describe the actual insufficiency, and update regex tests without changing model behavior | Implemented on current head; exact-head review follow-up required |
| The low-level all-missing MH-RM path was not directly covered while the production boundary correctly rejects it as unready | Preserve the diagnostic fitter's existing low-level behavior with a stub-core regression, while keeping `fit_irt_experiment` fail-closed on insufficient observations | Implemented on current head; exact-head review follow-up required |
| Exact-head review found raw-regex lint warnings, an unsorted `__all__`, and a missing citation paragraph boundary | Fix the minimal documentation/test hygiene issues, keep the public export set unchanged, and rerun the targeted suite before the next review cycle | Implemented on current head; exact-head review follow-up required |
| Strict Ruff review also classified invalid readiness controls, MH-RM seeds, and RSM category-count types as type errors rather than value errors | Raise `TypeError` for those invalid Python types, retain `ValueError` for numerically invalid values, and keep regression tests aligned with the public fail-closed contract | Implemented on current working tree; remote exact-head review follow-up required |
| The 2026-08-14 protection re-audit found fast-mlsirm's classic `main` protection reported `required_approving_review_count=0` and `require_last_push_approval=false`, while the rules endpoint also exposed an overlapping one-approval policy with no source identifier | Restore one independent current-head approval and last-push approval on fast-mlsirm, preserve stale-review dismissal and thread resolution, and verify classic protection, rules, PR aggregate state, and required checks after the mutation; never self-approve or merge through the ambiguity | Corrected 2026-08-14 via the protected API; classic protection now reports one approval and last-push approval, while the overlapping rules response remains an audit caveat |
| Current PR #816 jobs `31733222529`/`94558486829` and `31733230722`/`94558518535` remain queued with `labels=["ubuntu-latest"]` and empty `runner_name`; the repository runner API reports zero, but GitHub status reports Actions operational | Treat the queue as hosted-runner/org-capacity evidence until job assignment or a provider-specific failure proves otherwise; preserve the exact head and required contexts, do not cancel/status-retry, and do not classify it as a local MLX or LibreSSL defect | Observed 2026-08-14; active exact-head CI follow-up |
| A repeated Gemma 4 e4b `ModelClient.batch_chat` sweep through contextual-orchestrator used eight requests with server `prompt-concurrency=1`/`decode-concurrency=1`; client c=`1,2,4,8` averaged `2.095`, `2.083`, `2.092`, and `2.088` req/s over two runs | Keep the fast judge's bounded concurrency capability, but retain client c=1 as the default for this single-queue worker and retune only with a changed server queue/model/prompt budget; never infer judge quality from throughput | Observed 2026-08-14; contextual benchmark recorded, workload-specific retuning remains required |
| The current integrated Gemma 4 e4b anchored smoke completed four Boolean calls through the contextual adapter in `4.404 s` with `1,797` provider tokens and produced categories `release_monitoring=2`, `rollback_safety=2`, and polytomous row `[2,2]`; a one-criterion projection was rejected | Preserve the contextual-only Judge transport, require multiple criterion columns for IRT output, and retain scalar rejection as a contract invariant; do not pad, keyword-match, or infer an extra item | Verified 2026-08-14; semantic calibration remains separate from integration success |
| Fast PR #816 was `BEHIND` because protected `main` advanced from `fa23aaa` to `4f9276b` with the independent RAG/facet calibration change after the prior judge push. A clean worktree merged `origin/main` normally at `bbf5d0e1d1185d4a51fae24fa95c3c18a3ea2f23` without conflict; the integrated Python suite passed `3659` tests with two existing NumPy warnings. | Treat base synchronization as a required review boundary: merge the current protected `main` in an isolated clean worktree, preserve user-owned dirty files, rerun the full suite plus targeted judge/IRT checks, then push the named PR branch and invalidate all predecessor reviews/checks. Never merge a behind PR or reuse predecessor evidence. | Remediated locally 2026-08-14; exact-head remote review/check follow-up required |
| The latest exact-head re-audit found fast PR #816 at `e407d818ed4693cf2a725024ea505f0f1a82c695`, with the named branch and pull ref agreeing, but aggregate `REVIEW_REQUIRED`, no independent current-head approval, and 11 non-terminal queued CheckRuns. A representative `Analyze (actions)` job is queued with `labels=["ubuntu-latest"]` and an empty `runner_name`; the repository runner API reports zero runners while GitHub Actions reports operational. | Request a fresh authorized independent review after every push, wait for terminal same-head required checks and structured same-head Strix evidence, and perform a final protection/refetch audit before normal merge. Do not treat local suites, CodeRabbit `COMMENTED`, queued checks, or hosted-runner absence as approval or a clean security result. | Observed 2026-08-14; active protected-merge follow-up |
| A full-repository Ruff scan reports a large pre-existing baseline (705 findings) outside the current judge/IRT change set, while the reviewed IRT/Judge files pass the targeted `RUF022,RUF043` check. | Do not bulk-rewrite unrelated numerical or legacy modules in this PR. Inventory the baseline and add a separate scoped lint-burn-down/changed-file gate; keep the current PR's targeted lint evidence explicit so new review findings cannot be hidden by the baseline. | Observed 2026-08-14; separate maintenance direction |

## References

- Samejima, F. (1969), *Estimation of latent ability using a response pattern
  of graded scores*, https://doi.org/10.1007/BF03372160.
- Li, Q., et al. (2025), *Evaluating Scoring Bias in LLM-as-a-Judge*,
  https://arxiv.org/abs/2506.22316.
- Pezeshkpour, P., & Hruschka, E. (2024), *Large Language Models Sensitivity
  to The Order of Options in Multiple-Choice Questions*,
  https://aclanthology.org/2024.findings-naacl.130/.
- Jones, W. P., & Loe, S. A. (2013), *Optimal Number of Questionnaire Response
  Categories: More May Not Be Better*,
  https://doi.org/10.1177/2158244013489691.
- Muraki, E. (1992), *A generalized partial credit model: Application of an EM
  algorithm*, https://doi.org/10.1177/014662169201600206.
- Cai, L. (2010), *Metropolis-Hastings Robbins-Monro algorithm for confirmatory
  item factor analysis*, https://doi.org/10.3102/1076998609353115.
- Iannario, M., Monti, A. C., & Scalera, P. (2022), *The number of response
  categories in ordered response models*, https://doi.org/10.1515/ijb-2021-0013.
