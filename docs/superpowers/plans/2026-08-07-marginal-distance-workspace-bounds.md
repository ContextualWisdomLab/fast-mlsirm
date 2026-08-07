# Marginal Distance Workspace Bounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`, then `superpowers:verification-before-completion`. Keep this PR Draft until the exact unchanged head satisfies every repository gate.

**Goal:** Remove the unbounded three-dimensional item-by-node-by-dimension subtraction from the NumPy MMLE reference/fallback path, enforce an explicit float64 byte ceiling before node or distance allocation, and preserve stable Euclidean-distance semantics and parity with the Rust-first estimator.

**Architecture:** Add one checked byte-product helper, one pre-allocation validator, and one coordinate-subtraction-first pairwise-distance helper in `python/fast_mlsirm/estimators/marginal.py`. The helper keeps one output matrix plus one same-shaped reusable scratch matrix, avoiding both an `L × R × D` broadcast and the cancellation-prone squared-norm identity. The Rust backend remains the production numerical path. Python changes only the existing reference/fallback implementation.

## Global constraints

- No public API, result schema, model identity, dependency, workflow, lockfile, database object, or version change.
- No universal speed or memory claim.
- `MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES` is separate from the dominant EM element-count ceiling.
- Distance bytes are computed with checked dimension × float64-itemsize arithmetic before allocation.
- The direct helper accepts finite two-dimensional exact NumPy `float64`, C-contiguous operands with one shared positive latent width; no hidden dtype or layout conversion is permitted.
- Pairwise differences are formed coordinate-by-coordinate into one reusable two-dimensional scratch and squared in place before accumulation.
- Likelihood, missingness, initialization, iteration, multigroup, multilevel, zero-inflation, anchors, covariates, and return transport remain unchanged.
- Added production statement/branch coverage and public docstrings must be complete.

---

## Task 1 — RED byte and numerical contracts

**Files**

- Modify: `tests/test_marginal_distance_workspace_bounds.py`
- Reference: `python/fast_mlsirm/estimators/marginal.py`

- [x] Require exact pairwise-distance parity and zero/round-off behavior.
- [x] Require deterministic high-offset translation stability for both `L < R` and `L > R` against the direct coordinate-difference equation.
- [x] Require non-finite input rejection before the numerical kernel.
- [x] Require checked float64 byte accounting and overflow-safe giant-dimension rejection.
- [x] Require separate pairwise-output-plus-scratch and item-gradient byte gates.
- [x] Prove an otherwise-valid public spatial estimator request fails before `_xi_nodes`.
- [x] Require exact Boolean `uses_space`, valid dimensions/itemsize/limit, and finite positive epsilon.
- [x] Pin subtraction/square/sqrt `out=` reuse and the absence of matmul/einsum and the 3-D broadcast.

Run:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py
```

Expected before production implementation: RED because the high-offset regression exposes cancellation in the squared-norm/matmul identity.

---

## Task 2 — Checked byte products and preflight

**Files**

- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

- [x] Retain module-level `MAX_MARGINAL_WORKING_SET = 100_000_000` for the existing dominant element contract.
- [x] Retain `MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024` for distance allocations.
- [x] Implement `_checked_marginal_workspace_bytes(label, *dimensions, itemsize, limit_bytes)` using exact-integer validation and division-before-multiplication.
- [x] Preserve the existing dominant EM gate as a separate unit contract; do not change its accepted dimensions.
- [x] Implement `_validate_marginal_distance_workspaces` with exact-Boolean behavior, pairwise output-plus-scratch bytes, finite-mask phases, and intentional `n_x × latent_dim` derivative bytes.
- [x] Normalize model/rule/dimensions and invoke distance preflight before `_xi_nodes`.

Focused verification:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py -k 'workspace or public_estimator'
```

---

## Task 3 — Translation-stable pairwise-distance helper

**Files**

- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

- [x] Validate exact NumPy array type, rank, float64 dtype, C contiguity, shared positive latent width, finiteness, byte budget, and epsilon before arithmetic.
- [x] Allocate one zeroed `L × R` float64 output and one same-shaped scratch matrix.
- [x] For each latent coordinate, call `np.subtract(..., out=scratch)`, `np.square(scratch, out=scratch)`, and accumulate into the output.
- [x] Release scratch before output-finiteness validation; add epsilon and apply `np.sqrt(..., out=distances)` in place.
- [x] Do not use the cancellation-prone `||x||² + ||y||² - 2x·y` identity, a second persistent pairwise result matrix, or any `L × R × D` difference tensor.
- [x] Preserve deterministic C-order output and finite checks.

Focused verification:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py -k 'pairwise or translation'
```

---

## Task 4 — Rewire existing distance paths

**Files**

- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

- [x] Reuse the helper in `_build_tables`.
- [x] Reuse it for candidate item predictors.
- [x] Reuse it for the tau M-step.
- [x] Replace the covariate `x_grid[None, :, :] - zeta[:, None, :]` broadcast.
- [x] Retain the guarded `diff = x_grid - zeta_i[None, :]` required by the zeta derivative.
- [x] Preserve non-spatial paths without distance allocation.

Focused verification:

```bash
pytest -q \
  tests/test_marginal_distance_workspace_bounds.py \
  tests/test_estimator_marginal.py \
  tests/test_objective.py
```

---

## Task 5 — Realistic parity and allocation evidence

**Files**

- Modify: `tests/test_marginal_distance_workspace_bounds.py`
- Maintain: `benchmarks/benchmark_marginal_distance_workspaces.py`

- [x] Add a moderate partially observed deterministic estimator case and compare applicable returned finite arrays, likelihood trace, status, and iteration count across repeated identical runs.
- [x] Exercise a valid covariate path and prove the governed helper is used without changing fitted evidence.
- [x] Record the largest existing repository fixtures accepted by the 128 MiB private ceiling.
- [x] Maintain a safe command-line benchmark reporting Python, NumPy, BLAS, operating system, processor, dtype, dimensions, warm-ups, repetitions, elapsed distribution, and peak traced/RSS memory.
- [x] Compare with the former broadcast only below a conservative benchmark safety limit.
- [x] Describe all measurements as environment-specific; claim only removal of the named broadcast, elimination of the identified high-offset cancellation mechanism, and enforcement of the byte ceiling.
- [ ] Re-run the benchmark on the final stable-kernel head and replace predecessor-head measurements before Ready status.

---

## Task 6 — Doctoring and release-record parity

**Files**

- Modify: `docs/doctoring/marginal-distance-workspace-bounds.md`
- Modify: `docs/changelog.d/563-marginal-distance-workspace-bounds.md`
- Modify: `CHANGELOG.md`
- Modify: PR description

- [x] Record the cancellation limitation, coordinate-subtraction design, 128 MiB private budget rationale, allocation phases, precision treatment, compatibility evidence, failure modes, rollback, Rust ownership, and APA 7 references.
- [x] Update the authoritative fragment to describe the final stable kernel.
- [ ] Render the authoritative changelog with the repository renderer; do not hand-maintain a divergent block.
- [ ] Update the PR body from RED-only language to the final exact implemented behavior and unchanged-head evidence.

Commands:

```bash
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

---

## Task 7 — Exact-head verification and merge discipline

- [ ] Run focused tests first on the final implementation head.
- [ ] Run complete Python statement/branch coverage and public-docstring gates.
- [ ] Run Rust workspace/all-target tests and clippy/fmt.
- [ ] Run PyO3, wheel reinstall, package acceptance, explicit GPU-no-skip, and fuzz.
- [ ] Require Security Scan and SAST on the same unchanged head.
- [ ] Require current-head CodeRabbit/OpenCode/Noema/Strix feedback and a qualifying non-author approval.
- [ ] Require zero unresolved actionable threads.
- [ ] Remove `needs-revision` / `do-not-merge`, mark Ready, and enable auto-merge only after every gate is satisfied.
- [ ] Verify accepted `main`; close issue #563 only after protected merge and accepted-main evidence.
