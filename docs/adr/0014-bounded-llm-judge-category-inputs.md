# ADR-0014: Bounded LLM-judge category inputs and security-scan evidence

Status: **Proposed**
Date: 2026-08-12
Supersedes: none
Superseded by: none

## Context

The LLM-as-a-Judge adapter accepts category counts and category values at a
public Python boundary before projecting results into dichotomous or
polytomous IRT items. The current-head Strix run for PR #778 at
`0721e55ce5d889b7917aa7da2891367adf3430dc` reported a supposed integer
overflow in this path. Python's built-in integers do not wrap, so that claim
was not independently valid. The review did, however, expose a real adjacent
boundary issue: `isinstance(value, int)` admits an `int` subclass whose
comparison methods can return forged results, allowing the category-count
bound to be bypassed before a resource-amplifying operation.

The same trust-boundary concern applies to category values accepted by the
deterministic IRT projection. A model response parsed by `json.loads` contains
built-in scalar types, but callers can construct `LLMJudgeResult` directly.
Security evidence must therefore distinguish a real code defect from a model
or provider failure and must never weaken the fail-closed gate. Review of this
active PR also found that an unhashable `category_method` could leak a raw
`TypeError` during set membership instead of the package-owned `ValueError`.

## Decision drivers

- Reject untrusted or adversarial runtime scalar objects before comparisons,
  arithmetic, or allocations.
- Preserve the existing dichotomous/polytomous response-matrix contract.
- Keep LLM security findings advisory until independently reproduced by code,
  tests, or a structured scanner artifact.
- Keep provider rate limits separate from source-security conclusions.

## Ownership and dependency direction

`fast-mlsirm` owns the judge result contract and its IRT projection. LLM
transport remains injected through `contextual-orchestrator`; neither a model
response nor a Strix narrative becomes numerical or merge authority.

## Decision

1. `category_count` accepts only an exact built-in Python `int` in
   `2..MAX_JUDGE_CATEGORIES` (currently 64). Booleans, floats, subclasses,
   negative values, and oversized integers fail before any derived list or
   threshold array is materialized.
2. Direct category values accept only exact built-in `int` or `float` values;
   booleans, subclasses, non-finite values, fractional values, and out-of-range
   values fail closed.
3. Any cumulative-threshold result remains bounded by the validated category
   count, and every IRT row still requires multiple criterion items and passes
   the shared dichotomous/polytomous response-matrix validator.
4. A Strix/provider failure is recorded with its exact head SHA, run URL,
   failed step, backend error, and structured-report availability. A model-only
   claim is not called a confirmed vulnerability without independent evidence.
5. Required checks remain fail-closed. No scan failure is bypassed, and no
   self-approval or protected-branch merge is manufactured.
6. `category_method` must be an exact built-in string before vocabulary
   membership is checked; unsupported or unhashable values fail with the
   package-owned `ValueError`.
7. Public score, mode, item-type, text, criterion-key, and usage-counter
   boundaries reject runtime subclasses before invoking conversion hooks or
   set membership. Oversized numeric JSON values become `JudgeFormatError`
   rather than leaking `OverflowError`; non-built-in usage counters are
   ignored rather than compared or accumulated.
8. `JudgeCriterion.criterion_id` accepts only an exact built-in `str` before
   regex validation, hashing, dictionary-key construction, or category
   template generation; runtime string subclasses fail with the package-owned
   `ValueError`.
9. Strict structured-output failure remains a failed comparison. An identical
   second completion through `contextual-orchestrator` is not a repair contract:
   it may be measured, but the final response must pass the same parser and no
   keyword, positional, or silent-drop fallback may convert failure into an
   IRT category.

## Invariants / acceptance evidence

1. `tests/test_llm_judge.py::test_category_count_and_category_values_reject_runtime_subclasses`
   rejects booleans, floats, values above 64, astronomically large integers,
   and comparison-forging `int` subclasses.
2. `judge()` and `LLMJudgeResult.to_irt_row()` share the same bounded category
   validation before cumulative threshold allocation or IRT projection.
3. `tests/test_llm_judge.py::test_judge_rejects_unknown_category_method` covers
   unknown strings, lists, and dictionaries without leaking `TypeError`.
4. `tests/test_llm_judge.py::test_judge_rejects_overflowing_and_runtime_subclass_scores`
   and `test_judge_text_and_usage_boundaries_reject_runtime_subclasses` cover
   overflow, conversion-hook, unhashable-mode, item-type, text, and usage
   boundary behavior.
5. `tests/test_llm_judge.py::test_judge_rejects_unhashable_criterion_id_before_category_template`
   rejects an unhashable `str` subclass before it can become a category
   template key or leak a raw `TypeError`.
6. PR #778 Strix run `31549881616` is immutable evidence for exact scan head
   `c54706cc16c8452d603c22d9604e7d27ede6288f`:
   `https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/31549881616`
   and job
   `https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/31549881616/job/93970141054`.
   GitHub recorded no failed job step; `Run Strix (quick)` itself concluded
   success. Inside that step, however, `gate-console.log` recorded NVIDIA NIM
   HTTP 429 rate limiting, GitHub Models HTTP 410 retirement-brownout failures,
   repeated fail-closed/no-report markers, and a generic report that admitted an
   incomplete AST pass. Artifact `9124255508` (`strix-reports`) therefore
   predates the trusted provenance contract and has no authoritative
   `evidence-binding.json`. This is provider-degraded/incomplete evidence, not
   a clean security result.
7. PR #778 Strix run `31552408884` is immutable evidence for exact scan head
   `6c42d4a53d6d70cb1ae0127df624c3cc178ddd4b`:
   `https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/31552408884`
   and job
   `https://github.com/ContextualWisdomLab/fast-mlsirm/actions/runs/31552408884/job/93977765693`.
   Again, GitHub recorded no failed job step and `Run Strix (quick)` concluded
   success, while `gate-console.log` recorded NVIDIA NIM HTTP 429, GitHub Models
   HTTP 410 retirement-brownout, fail-closed/no-report markers, and a generic
   fallback report. Artifact `9125226971` (`strix-reports`) lacked
   `evidence-binding.json`, and its successful `run.json` omitted authoritative
   head metadata. It is inconclusive provider evidence, not a clean scan.
8. The central correction is owned by `ContextualWisdomLab/.github` PR #937,
   exact active-PR head `2c6f4323ac864587d767824464379678ebfe888a`.
   Its `.github/workflows/strix.yml` change removes provider-outage
   neutral-success behavior, records the exact scan-start head, requires a
   completed successful `run.json` plus non-empty report, rejects fail-closed
   provider markers, and emits `evidence-binding.json` bound to the exact head
   and report digest. Protected central `main` is still
   `6eb06cdd08c79a06f7b390069d4ffa49e2eb7dba`, so #937 is a read-only external
   prerequisite here rather than shipped central truth. After it integrates,
   this PR must be rescanned on its then-current exact head; no predecessor
   Strix result transfers.
9. The full Python and Rust test suites, targeted Ruff checks, and exact-head
   required checks must pass before this hardening is considered integrated.

10. A 2026-08-12 local 3B probe used the
    `ContextualOrchestratorJudge -> contextual-orchestrator -> mlx-lm` route
    with temperature `0`, disabled thinking, and 256 output tokens. Nine
    direct K-way calls parsed but varied across K and framing: neutral
    `(0.0000, 1.0000, 0.8333)`, liked `(0.0000, 1.0000, 0.9167)`, and
    disliked `(0.0000, 0.7500, 1.0000)` for K=`(2,5,7)`. Four cumulative
    threshold calls failed strict parsing, and one identical retry per call
    recovered none. This is calibration evidence, not a quality or bias
    conclusion.

11. Central PR #937's latest green Strix job
    `31555003423`/`93985504528` for head
    `2c6f4323ac864587d767824464379678ebfe888a` did not execute the PR-head
    workflow definition: its step list had no `Validate Strix report
    provenance` step, its artifact had no `evidence-binding.json`, and its
    log contained the old neutral provider-outage skip plus NVIDIA 429 and
    GitHub Models 410 failures. This is base-workflow evidence, not proof of
    the central PR change; after central integration, this PR must be rescanned
    through the active trusted workflow at its exact head.
12. Central PR #937 then advanced to exact head `8726df15` with a separate
    non-privileged `strix-workflow-contract.yml` data-only workflow and a
    doctoring record for the base-workflow evidence boundary. All previous
    central checks and review interpretations are stale for this new head.
    The provider-backed provenance binding remains unproven until central
    integration and a fresh exact-head run emit a binding manifest without
    provider-failure markers.
13. A review regression showed that a caller-supplied `list` subclass in the
    orchestration trace could execute overridden iteration/length hooks during
    usage aggregation and trace-count reporting. The judge now accepts only
    the exact built-in list for provider trace accounting; subclass traces
    produce zero derived usage and zero trace steps without executing hooks.

14. CodeRabbit's exact-head review of central PR #937 at `8726df15` found that
    the new data-only Strix workflow contract trusted marker strings: a phrase
    could be supplied only by a comment or by a statically unreachable
    `if: false` step. The contract now parses the fetched workflow with
    `Psych.safe_load`, follows the reachable `jobs.strix.steps` structure and
    required order, and checks executable provenance commands plus the
    fail-closed gate. Comment-only and unreachable fixtures are regression
    cases. This is still pre-integration evidence; all prior central Strix
    results remain stale until the exact head is reviewed and a post-integration
    run emits a clean binding manifest.

15. A live bearer-authenticated probe used the configured
    `mlx-community/llama-3.2-3b-instruct-4bit` worker through
    `ContextualOrchestratorJudge -> contextual-orchestrator -> mlx-lm`. On the
    same answer and two criteria with `category_count=5`, direct judging
    produced score `1.0000` (`4/4`, accepted) and cumulative-threshold judging
    produced score `0.0000` (`0/0`, rejected). Both outputs passed the same
    strict parser and retained one nested contextual trace step (568 and 570
    tokens respectively). This is paired method sensitivity, not a claim that
    more choices cause positive bias; method, K, trace, usage, and parse status
    remain part of the calibration denominator.

## Non-goals and claims not made

- This decision does not claim that Python integers overflow or that a single
  model-generated security report proves a CVSS score.
- Exact scalar typing does not establish that LLM judgments are unbiased.
- This decision does not authorize keyword matching, direct provider calls, or
  a single scalar response as an IRT dataset.

## Consequences and trade-offs

### Benefits

- Resource-amplifying category bounds cannot be bypassed through comparison
  hooks or coercion protocols.
- Security and provider-availability evidence remains auditable and separate.
- The public judge path stays deterministic and compatible with the existing
  multi-item IRT contract.

### Costs / risks

- Legitimate custom numeric subclasses are rejected and must be normalized by
  the caller before crossing the boundary.
- Strix runs remain dependent on external model quotas and can require a
  rerun after a provider outage.

## Alternatives considered

### Coerce every value with `int()` or `float()`

Rejected because conversion hooks can execute caller code and can silently
change values at a security and measurement boundary.

### Trust the Strix narrative as the vulnerability oracle

Rejected because the observed run lacked a structured report, began with a
provider 429, and made a false claim about Python integer wraparound.

### Ignore the finding because the primary claim was false

Rejected because the same code admitted adversarial `int` subclasses that could
forge comparisons. The boundary is now hardened and regression-tested.

### Blindly retry or repair invalid model output

Rejected as a default because the local probe recovered no cumulative-threshold
failure after one identical contextual-orchestrator retry, and any lexical or
positional repair would violate the semantic judge boundary. A future
independent binary-threshold decomposition or stronger local judge must be
benchmarked with added cost, first/final parse status, held-out paired cases,
and the same fail-closed parser.

### Iterate over arbitrary trace containers

Rejected because a provider-controlled or caller-supplied list subclass can
execute code through `__iter__` or `__len__` while the result record is being
assembled. Exact built-in container typing keeps usage and trace metadata
bounded and side-effect free; malformed custom traces are treated as absent.

## Failure, degraded, and recovery behavior

Malformed category, score, text, mode, item-type, and criterion-key inputs raise
the package-owned `ValueError` or `JudgeFormatError` before model output is
accepted or an IRT row is returned. Invalid usage counters are deliberately
ignored by `_usage()` and contribute zero token totals; they are not promoted
to score, category, or merge evidence.
Malformed model JSON, missing structured scan evidence, provider rate limits,
and failed required checks remain fail-closed. After a code change, rerun the
security workflow against the exact pushed head and record the new result;
never reinterpret a stale run as evidence for a new head.

## Security and privacy implications

The change reduces coercion and resource-exhaustion risk without adding
credentials or retaining source text. Strix logs and provider errors may be
retained as CI evidence, but secrets must remain masked. The exact head SHA
and run URL are required for reproducibility; they do not grant merge
authority.

## Compatibility, migration, and rollback

The accepted JSON schema and valid category range are unchanged. Callers that
used custom numeric subclasses must pass built-in scalars. Rollback is allowed
only with a superseding ADR and equivalent boundary tests; reverting to
coercive validation is not an acceptable compatibility path.

## Verification and release evidence

- Run the targeted judge tests and Ruff on changed Python files.
- Run the full Python suite, Rust workspace tests, and binding tests.
- Re-run Strix and all required checks on the exact pushed PR head.
- Review current inline/general PR feedback and obtain any approval required by
  the live repository policy before a protected merge.
- Preserve contextual-orchestrator routing for every LLM-as-a-Judge call.
- Keep the 3B retry probe and any future threshold-decomposition comparison in
  the calibration denominator; do not promote a recovered subset to an IRT
  release claim without paired gold/human evidence.

## Research and standards basis

Samejima, F. (1969). *Estimation of latent ability using a response pattern of
graded scores*. Psychometrika, 34(S1), 1–97. https://doi.org/10.1007/BF03372160.
The Psychometric Society provides an openly accessible reproduction at
https://www.psychometricsociety.org/sites/main/files/file-attachments/mn17.pdf.
Samejima's graded-response formulation uses ordered cumulative boundaries for
polytomous categories, which is the limited psychometric basis for representing
our cumulative threshold vector. It does not validate LLM prompts, rater
fairness, equal weighting, or the claim that more response options cause
positive bias; those remain empirical calibration questions. The local Zotero
record is parent `XR8LNVF5` with the 97-page PDF attachment `345PA99V`
(37,178,609 bytes; MD5 `e558448c14a2e400a947e19f16cbcb7e`).

## Follow-ups

- Re-run PR #778's Strix workflow after the central trusted-gate hardening and
  compare the exact head SHA, provider status, structured artifact, and finding
  intersection; require a current-head report/provenance binding.
- If a later scan reports a reproducible issue, create a superseding ADR or
  append a new evidence record with the reproducer and regression test.

## Reversal / supersession conditions

Supersede this ADR only if the public judge contract adopts a different
bounded scalar representation or if structured security evidence demonstrates
that the current exact-type boundary is insufficient.
