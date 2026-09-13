# ADR-0029: Use a fixed exact accumulator for correctly-rounded finite binary64 means

Status: **Proposed**  
Date: 2026-09-11  
Supersedes: none  
Superseded by: none

## Context

`fast-mlsirm` needs a reusable, domain-neutral arithmetic mean primitive for finite IEEE-754 binary64 evidence. Downstream TEPP consumer evidence in `ContextualWisdomLab/TEPP#310` demonstrates an ordinary three-value mixed-sign case where pair-order heuristics, simple compensation, and same-side coalescing return a result one ULP above the correctly-rounded exact mean. The reusable numerical repair belongs here; temporal/event composition remains TEPP-owned.

The contract must preserve each admitted binary64 addend exactly until the final projection, distinguish exact cancellation from nonzero underflow that rounds to zero, remain permutation deterministic, avoid intermediate floating overflow when the mean is representable, and place no arbitrary psychometric sample ceiling on callers.

## Decision drivers

- Correctly-rounded round-to-nearest, ties-to-even output from the exact represented inputs and original slice cardinality.
- Determinism across input permutation and any future worker decomposition.
- Explicit proof that every materializable slice on supported Rust targets fits the accumulator.
- Rust-owned production arithmetic with no Python numerical fallback.
- A small, auditable implementation whose failure modes are bounded before release.
- A stable versioned public numerical contract that downstream repositories consume only after immutable release.

## Ownership and dependency direction

`fast-mlsirm` owns the domain-neutral binary64 numerical primitive in the Compute Backend / Public Binding boundary. It does not own TEPP event-time semantics, aggregation policy, missingness policy, longitudinal estimands, or orchestration. TEPP may consume only an immutable released/versioned contract; mutable PR-head pinning, source copying, cross-service SQL, and a TEPP-local arithmetic fork are prohibited.

The decision introduces no contextual-orchestrator, provider, database, identity, or hosted-service dependency.

## Decision

Represent every admitted finite binary64 value exactly as a signed integer multiple of `2^-1074`. Accumulate positive and negative magnitudes separately in fixed-width 34×`u64` unsigned magnitudes, subtract the smaller exact magnitude from the larger once, divide the resulting exact magnitude by the original slice cardinality using integer quotient/remainder, and perform exactly one final binary64 round-to-nearest, ties-to-even projection.

The public contract is `fast_mlsirm.binary64_mean@1.0.0`, implemented by `mlsirm_core::binary64_mean::correctly_rounded_finite_mean`. Empty input and NaN/infinity fail closed. Exact cancellation is canonicalized to `+0.0` with `exact_zero=true`; a nonzero exact mean that rounds to zero retains `exact_zero=false` and preserves a negative-zero sign for negative underflow.

This ADR remains Proposed until the implementation lands on the protected branch with the verification and immutable-release evidence below. It must not be marked Accepted merely because the PR implementation or a review model reports success.

## Invariants / acceptance evidence

1. Every finite binary64 addend is represented exactly before accumulation. A normal coefficient requires at most 2,098 magnitude bits in `2^-1074` units.
2. Supported targets enforce `usize::BITS <= 64`; any materializable same-sign slice sum is `< 2^2162`. Thirty-four `u64` limbs provide 2,176 magnitude bits, leaving a proved capacity margin without a domain sample ceiling.
3. Positive/negative totals are reduced exactly before division by the original count; no floating reduction tree is part of the public contract.
4. The original-count quotient/remainder participates in final rounding. Discarded quotient bits and a nonzero remainder provide the sticky information needed to distinguish below-half, above-half, and exact-half cases.
5. Exact-half cases use ties-to-even; subnormal-to-minimum-normal and normal binade carries are explicit tested transitions.
6. Public regression evidence includes the TEPP half-ULP case and mirror, `[1e16,-1,-1]`, `[MAX,1e-16,-MAX]`, `[MAX,2*MIN_SUBNORMAL,2*MIN_SUBNORMAL,-MAX]`, same-sign `MAX`, exact cancellation, signed nonzero underflow, subnormal/normal boundaries, permutation invariance, and non-finite admission failure.
7. Repository-owned deterministic property evidence compares a broad subnormal domain against an independent `i128` exact integer/rational oracle. External model review is supplementary and cannot replace repository tests.
8. Before landing, the exact current head must have Rust tests, rustdoc/Clippy, owned production statement and branch coverage, package/fuzz/security evidence, zero valid unresolved findings, and a qualifying independent approval. Predecessor-head evidence does not transfer after a push.

## Non-goals and claims not made

- This primitive does not define psychometric estimands, temporal aggregation, missingness, sampling, weighting, or scientific recovery acceptance.
- It does not claim arbitrary-precision public arithmetic; the fixed width is deliberately proved from the complete finite-binary64 domain and supported slice cardinality.
- It does not authorize an f32/GPU approximation as evidence-equivalent to the deterministic CPU-f64 reference.
- It does not by itself establish true-parameter recovery for LSIRM/MLSIRM/IRT estimators; estimator-level scientific acceptance remains with those owning model lanes.
- It does not expose an exact sum object or parallel accumulator API in version 1.0.0.

## Consequences and trade-offs

### Benefits

- Correct rounding is independent of input order and ordinary floating reduction behavior.
- Cancellation can reveal low-order mass that compensation/coalescing heuristics can lose.
- Same-sign near-`f64::MAX` inputs do not overflow merely because an intermediate floating sum would overflow.
- Memory is fixed and small: two 2,176-bit magnitudes plus bounded scalar state, independent of input length.
- The representation can support a future associative parallel accumulator without changing numerical semantics, provided merge/finalization receives its own contract and parity evidence.

### Costs / risks

- The scalar fixed-limb path is more code than a native `f64` reduction and may be slower for small slices; correctness, not speculative speed, is the release criterion for this primitive.
- Hand-written carry, borrow, division, and final-rounding logic creates a concentrated correctness surface that requires exact-oracle, boundary, fuzz, and coverage evidence.
- Future support for targets with `usize::BITS > 64` would invalidate the current cardinality proof and requires a superseding capacity decision before admission.

## Alternatives considered

### Naive, pairwise, sorted, or pair-order floating summation

Rejected for the public correctly-rounded contract. A deterministic reduction order can still round intermediate partial sums; sorting or pairing by magnitude does not preserve every represented addend and the TEPP counterexample demonstrates that a plausible heuristic can remain one ULP wrong.

### Kahan/Neumaier compensation and same-side coalescing

Rejected as the canonical contract. Compensation substantially improves ordinary summation error but is not an exact accumulator over the complete binary64 exponent range and does not provide a direct correctly-rounded final-mean proof after arbitrary mixed-sign cancellation.

### Error-free transformations / floating partial expansions

Considered seriously because Ogita, Rump, and Oishi (2005) and Rump, Ogita, and Oishi (2008a, 2008b) establish accurate, faithful, and rounding-oriented summation techniques. Deferred for this API because a complete expansion/underflow/final-division proof would be more complex than the fixed exact representation once the finite-binary64 and 64-bit-cardinality bounds are used. These methods remain valid future implementation candidates only if they preserve the same public exact-rounding invariants and pass the same oracle evidence.

### Arbitrary-precision integer/rational production dependency

Rejected for production. It would satisfy exactness but adds an avoidable runtime and supply-chain dependency when the complete admitted domain has a small provable fixed representation. Arbitrary-precision or exact rational arithmetic remains appropriate as independent test/review evidence.

### Pre-scaling or normalization before accumulation

Rejected. Scaling can erase subnormal residue that becomes material after cancellation and would make exact-zero versus rounded-zero classification dependent on an earlier rounding step.

## Failure, degraded, and recovery behavior

Empty slices and non-finite inputs fail closed with typed errors. There is no approximate fallback: an internal invariant failure, capacity-proof violation on a future unsupported target, or failed exact-current-head quality gate blocks release rather than selecting native floating summation.

Rollback is version-level. Until an immutable release is published, downstream consumers must keep their existing behavior and RED evidence rather than pin this branch. If a released defect is found, downstream migration stops or rolls back to the prior immutable contract while a superseding fast-mlsirm release repairs the numerical owner.

## Security and privacy implications

The primitive accepts only in-memory finite `f64` slices and creates no credential, network, persistence, PII, or provider boundary. Its security-relevant concerns are bounded work, denial-of-service resistance through fixed auxiliary memory, native arithmetic safety, and supply-chain integrity of the released crate/package. Release still requires the repository security/SBOM/provenance controls.

## Compatibility, migration, and rollback

The initial public identity is `fast_mlsirm.binary64_mean@1.0.0`. No downstream repository may consume the mutable PR head. After protected integration and immutable release, TEPP may replace its local failing reduction through the released contract and must retain its consumer counterexample as migration evidence. A later change to admission, zero identity, rounding mode, target-width bound, or accumulator merge semantics requires a versioned contract change and a superseding ADR when it changes this decision.

## Verification and release evidence

Before this ADR can become Accepted and the contract can be released:

- the public directed fixtures and repository-owned exact integer/rational oracle must be GREEN on one unchanged exact head;
- private carry/borrow/bit-classification tests and fuzz evidence must be GREEN;
- owned production rustdoc, statement coverage, branch/edge coverage, Clippy, package and security/supply-chain gates must meet repository policy without skip/xfail/source rewriting or denominator tricks;
- current-head independent review must report no valid unresolved numerical finding;
- protected integration must be followed by version/changelog/tag/package publication with immutable release, SBOM, provenance, reproducibility and rollback evidence;
- TEPP migration occurs only after that release and remains a downstream consumer action.

## Research and standards basis

Institute of Electrical and Electronics Engineers. (2019). *IEEE standard for floating-point arithmetic* (IEEE Std 754-2019).

International Organization for Standardization. (2020). *Information technology—Microprocessor systems—Floating-point arithmetic* (ISO/IEC 60559:2020).

Ogita, T., Rump, S. M., & Oishi, S. (2005). Accurate sum and dot product. *SIAM Journal on Scientific Computing, 26*(6), 1955–1988. https://doi.org/10.1137/030601818

Rump, S. M., Ogita, T., & Oishi, S. (2008a). Accurate floating-point summation part I: Faithful rounding. *SIAM Journal on Scientific Computing, 31*(1), 189–224. https://doi.org/10.1137/050645671

Rump, S. M., Ogita, T., & Oishi, S. (2008b). Accurate floating-point summation part II: Sign, K-fold faithful and rounding to nearest. *SIAM Journal on Scientific Computing, 31*(2), 1269–1302. https://doi.org/10.1137/07068816X

IEEE P754 is an active revision project as of 2026-09-11; it is not treated as a published replacement for the 2019/2020 contract in this decision.

## Follow-ups

- #1814 remains the owner issue until protected integration and immutable release acceptance are complete.
- #1816 is the implementation vehicle and remains Draft while exact-current-head evidence is incomplete.
- TEPP #310 remains the downstream consumer RED and must not copy this implementation.
- If the repository ADR queue allocates ADR-0029 concurrently before this PR lands, non-force reconcile this decision to a fresh conflict-free number; do not overwrite a sibling decision or force-rewrite history.

## Reversal / supersession conditions

Create a superseding ADR if the supported target cardinality invalidates the 2,176-bit bound, the public contract adopts a different rounding mode or exact-sum surface, a verified counterexample disproves correct rounding, or an alternative implementation can meet the same exact semantics with materially better audited performance/maintainability. Do not silently weaken the contract to faithful or approximately reproducible summation.
