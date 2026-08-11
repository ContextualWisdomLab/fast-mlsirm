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

## Invariants / acceptance evidence

1. `tests/test_llm_judge.py::test_category_count_and_category_values_reject_runtime_subclasses`
   rejects booleans, floats, values above 64, astronomically large integers,
   and comparison-forging `int` subclasses.
2. `judge()` and `LLMJudgeResult.to_irt_row()` share the same bounded category
   validation before cumulative threshold allocation or IRT projection.
3. `tests/test_llm_judge.py::test_judge_rejects_unknown_category_method` covers
   unknown strings, lists, and dictionaries without leaking `TypeError`.
4. PR #778's Strix log records an initial NVIDIA NIM 429, a fallback-model
   report, no structured vulnerability artifact, and the exact finding. The
   finding is treated as an adjacent hardening signal, not as proof of integer
   wraparound.
5. The full Python and Rust test suites, targeted Ruff checks, and exact-head
   required checks must pass before this hardening is considered integrated.

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

## Failure, degraded, and recovery behavior

Malformed category inputs raise the package-owned `ValueError` or
`JudgeFormatError` before model output is accepted or an IRT row is returned.
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
- Review current inline/general PR feedback and obtain an independent approval
  before a protected merge.
- Preserve contextual-orchestrator routing for every LLM-as-a-Judge call.

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

- Re-run PR #778's Strix workflow after this hardening and compare the exact
  head SHA, provider status, structured artifact, and finding intersection.
- If a later scan reports a reproducible issue, create a superseding ADR or
  append a new evidence record with the reproducer and regression test.

## Reversal / supersession conditions

Supersede this ADR only if the public judge contract adopts a different
bounded scalar representation or if structured security evidence demonstrates
that the current exact-type boundary is insufficient.
