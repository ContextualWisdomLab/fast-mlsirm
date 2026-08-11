# Bounded LSR Ranking Input Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:test-driven-development and superpowers:systematic-debugging. Implement tasks in order and keep the PR Draft until the complete exact-head gate is green.

**Goal:** Make the public LSR/I-LSR ranking wrappers fail closed on hostile, infinite, or oversized caller-controlled ranking iterables before NumPy allocation or Rust invocation, while preserving accepted ranking bytes and all Rust-owned numerical semantics.

**Architecture:** Keep one shared Python validation/CSR boundary in `python/fast_mlsirm/scaling.py`. Replace unbounded `list(ranking)`/Python-list accumulation with bounded streaming into fixed-width unsigned storage. Budget the live CSR payload explicitly from a private byte ceiling. Return contiguous `uint64` NumPy views/arrays to the unchanged Rust bindings. No ranking/scaling arithmetic moves into Python.

**Tech Stack:** Python 3.10+, NumPy, standard-library fixed-width storage, existing PyO3/Rust LSR kernels, pytest.

## Non-negotiable boundaries

- Work only in `ContextualWisdomLab/fast-mlsirm`.
- Do not modify Rust LSR/ILSR formulas, stationary-distribution logic, public signatures, result types, dependencies, workflows, model names, version, or release.
- `n <= 10_000` remains the dense-chain item ceiling.
- A single ranking can contain at most `n` entries; consume at most `n + 1` to prove overlength.
- Bound combined flattened item entries plus CSR start offsets before allocation using one documented private byte ceiling.
- Normalize ordinary caller-controlled iteration/callback failures to stable non-reflective `ValueError` messages. Preserve `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`.
- Preserve valid list, tuple, and generator inputs and byte-identical `uint64` values passed to Rust.
- Add 100% changed production statement/branch coverage and complete docstrings.

### Task 1: Establish fail-first resource and callback contracts

**Files:**
- Create: `tests/test_scaling_ranking_input_bounds.py`
- Modify later: `python/fast_mlsirm/scaling.py`

- [ ] Confirm the committed tests fail on protected main specifically because `_rankings_to_csr` performs unbounded `list(ranking)`/outer iteration, ignores a CSR byte ceiling, and leaks ordinary iterable exceptions.
- [ ] Confirm RED occurs quickly through finite probe iterables that raise if the implementation asks for more than the permitted bounded number of values; do not add an actually infinite CI test.
- [ ] Pin boundary-minus-one, exact-boundary, and boundary-plus-one behavior using a monkeypatched private byte ceiling so tests allocate only tiny fixtures.
- [ ] Pin propagation of process-control exceptions and redaction of ordinary exception text.
- [ ] Pin accepted list/tuple/generator numerical parity through the public Rust-backed `lsr_rankings` wrapper.

### Task 2: Implement bounded streaming CSR construction

**Files:**
- Modify: `python/fast_mlsirm/scaling.py`
- Test: `tests/test_scaling_ranking_input_bounds.py`

- [ ] Add a private `MAX_RANKING_CSR_BYTES` ceiling with a beginner-readable comment describing exactly which live fixed-width arrays it covers and which process-memory claims it does not make.
- [ ] Validate `n` before consuming caller iterables.
- [ ] Iterate each ranking with an explicit `n + 1` cap; validate each item as it is read; reject shorter-than-two and overlong rankings without materializing arbitrary iterables.
- [ ] Accumulate flattened indices and start offsets in fixed-width unsigned storage rather than retained Python-int lists.
- [ ] Before each append, use division-before-multiplication or equivalent checked arithmetic to prove `(flat_count + start_count) * 8 <= MAX_RANKING_CSR_BYTES` without oversized intermediate products.
- [ ] Convert or expose the fixed-width storage as contiguous `np.uint64` arrays/views without an unbudgeted second full-size copy.
- [ ] Catch ordinary iteration/callback exceptions at both outer and inner boundaries and raise stable `ValueError` without rejected values or exception text; do not catch process-control exceptions.
- [ ] Keep duplicate-item enforcement compatible with the current Rust contract unless bounded Python validation can reject earlier without changing accepted inputs.

### Task 3: Verify and document the boundary

**Files:**
- Modify: public LSR/I-LSR docstrings in `python/fast_mlsirm/scaling.py`
- Create: `docs/doctoring/lsr_ranking_input_bounds.md`
- Create: `docs/changelog.d/612-lsr-ranking-input-bounds.md`
- Modify: `CHANGELOG.md` through the authoritative renderer

- [ ] Document the input/resource boundary, stable failure semantics, Rust numerical ownership, and absence of universal memory/capacity claims.
- [ ] Run focused tests, existing scaling/LSR tests, Python branch coverage for the changed boundary, full Python, Rust/PyO3, package/reinstall/release acceptance, GPU-no-skip, fuzz, Security Scan, and SAST on one unchanged exact head.
- [ ] Request fresh exact-head automated review; resolve only valid addressed findings; keep Draft until every required gate and repository approval policy passes.

## Acceptance evidence

The slice is complete only when:

1. finite probes prove the implementation never asks an inner ranking for more than `n + 1` entries and never consumes an unbounded outer stream beyond the explicit CSR budget;
2. ordinary hostile iterable text cannot appear in public exceptions;
3. process-control exceptions still propagate;
4. boundary tests pass without large allocations;
5. accepted list/tuple/generator inputs produce numerically identical Rust-backed results;
6. the changed production boundary has 100% statement/branch coverage and public docs; and
7. the exact unchanged head passes the full repository merge contract.
