# Product and technical gap live refresh — 2026-09-11

Status: **Non-authoritative live supplement**  
Protected-product authority: `main@493326f2de49ea1704da0ded19868ed05d2fe00f`  
Central workflow authority: `ContextualWisdomLab/.github@cb0872c9a20d5584703dffacca65c096fc034c6c`  
Canonical historical baseline: `docs/product-technical-gap-baseline.md`  
Latest immutable product release: `v0.9.1` (published 2026-08-26)

This supplement supersedes only the live execution statements that changed after the 2026-09-10 refresh. Historical supplements remain evidence and are not rewritten. No Draft, predecessor check, foreign-owner PR head, or byte-identical predecessor tree is promoted to shipped status.

## Marginal binary64 contract: third stale-cancellation replay

`fast-mlsirm#1742` remains the canonical numerical-contract lane. The reviewed behavioral tree at `785e98c9282a51f77559e83133f5d8da16ad5942` contained exactly two protected-main deltas: `tests/test_marginal_distance_reduction_contract.py` and `tests/test_marginal_reduction_reproducibility.py` (+158/-0). Those tests preserve the exact finite one-ULP counterexample, exercise the real `fit_marginal_numpy` M-step, and show that the rejected row-wise reassociation changes returned zeta binary64 bits without modifying production estimator arithmetic.

A third obsolete cancellation then appended `80caf161dafa236b5e79beab0ce2fac11917a4da` with message `⚡ Bolt: 취소된 최적화 빈 커밋`. Despite the message, this was not tree-empty relative to its parent: its tree became `7485a3fef3f84ad3915bc75504b089fd930ee81f`, exactly the protected-main tree, so both still-valid unmerged regression files disappeared and the PR again became instantaneously zero-diff. No verified successor had inherited the scientific delta.

The lane was repaired without force update, destructive rebase, or predecessor-status transfer. Ordinary forward commits `e76868fe3ad42f404c3912445f72b7b2ac2fd42c` and `a6e09c61b2dc2a02e68a51a5b67d35d80f8a3e3b` restored the two files byte-for-byte. Fresh comparison `785e98c9282a51f77559e83133f5d8da16ad5942...a6e09c61b2dc2a02e68a51a5b67d35d80f8a3e3b` has no changed files, while protected main to current head is again exactly two files (+158/-0). #1742 stays Draft. New current-head checks are reacquired from scratch; predecessor GREEN is diagnostic evidence only.

The causal lifecycle owner remains `ContextualWisdomLab/.github#2082`. Automatic cleanup must inspect parent-to-head mutation and validated lineage/successor authority, not trust a commit message or terminal `base...head` zero-diff alone. A previously validated unmerged delta that disappears without a verified successor must fail closed to Draft/needs-repair. The contrasting #1805 case remains legitimately closable because its optimization was affirmatively rejected and no unique valid delta remained.

## Review gateway: successful admission followed by runtime 502

The product-gap lane `fast-mlsirm#1519@3bafc62cf98c9c6dacc1afff0d299626f888330a` reached GREEN ordinary CI, repository CodeQL, Required CodeQL PR, Security Scan, Semgrep, bootstrap and coverage checks, but Required Noema and OpenCode failed outside the documentation source lane. The Required Noema job used trusted `.github@cb0872c9a20d5584703dffacca65c096fc034c6c`, which still vendored `contextual-orchestrator@414f22973658c4ddc3d4320fcf7acd9b4e8ba991` and injected raw provider credentials into the central consumer.

Preflight produced 24 candidates with `ready_count=3`, `deferred_count=4`, and `rejected_count=9`, and gateway preflight itself succeeded. The caller then made exactly one `orchestrator/free` request; after about 610.1 seconds it failed with HTTP 502 in `response_error`, served model `google/gemma-4-31b-it`. No caller retry multiplication and no publishable Noema source verdict occurred.

This is owner-path RED, not a reason to add leaf retries or provider-specific fallback. Evidence was handed to `contextual-orchestrator#1106`, exact Draft prerequisite `#1124@e3482266658ed871476c98cc720ebeabc66ae3da`, and central consumer migration `ContextualWisdomLab/.github#1759`. The acceptance sequence remains executable CO admission/runtime-routing contract → protected integration → immutable CO release → exact `.github` gateway-bearer-only consumer bump using `orchestrator/free` → unchanged leaf replay → removal of legacy raw-provider-secret/probing bootstrap.

## Landing authority

#1791 remains an otherwise-clean docs-only landing candidate at `bfe9c22ac4355da45504211148bec914bc632909`, but no qualifying independent current-head `APPROVED` review has been established. #1519 also has no merge authority while its current-head Required Noema/OpenCode review path is RED. Self-approval, routine administrator bypass, synthetic status, source-neutral retrigger, mutable foreign-owner pin, force update, destructive rebase, sample/tolerance weakening, and predecessor-success transfer remain inadmissible.
