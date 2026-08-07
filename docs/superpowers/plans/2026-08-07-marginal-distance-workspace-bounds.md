# Marginal Distance Workspace Bounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`, then `superpowers:verification-before-completion`. Keep this PR Draft until the exact unchanged head satisfies every repository gate.

**Goal:** Remove the unbounded three-dimensional item-by-node-by-dimension subtraction from the NumPy MMLE reference/fallback path, enforce an explicit float64 byte ceiling before node or distance allocation, and preserve parity with the Rust-first estimator.

**Architecture:** Add one checked byte-product helper, one pre-allocation validator, and one in-place squared-norm pairwise-distance helper in `python/fast_mlsirm/estimators/marginal.py`. The Rust backend remains the production numerical path. Python changes only the existing reference/fallback implementation.

## Global constraints

- No public API, result schema, model identity, dependency, workflow, lockfile, database object, or version change.
- No universal speed or memory claim.
- `MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES` is separate from the dominant EM element-count ceiling.
- Distance bytes are computed with checked dimension × float64-itemsize arithmetic before allocation.
- The direct helper accepts finite two-dimensional float64 matrices only; no hidden dtype-conversion copy.
- Likelihood, missingness, initialization, iteration, multigroup, multilevel, zero-inflation, anchors, covariates, and return transport remain unchanged.
- Added production statement/branch coverage and public docstrings must be complete.

---

## Task 1 — RED byte and numerical contracts

**Files**

- Modify: `tests/test_marginal_distance_workspace_bounds.py`
- Reference: `python/fast_mlsirm/estimators/marginal.py`

- [x] Require exact pairwise-distance parity and zero/round-off behavior.
- [x] Require non-finite input rejection before BLAS.
- [x] Require checked float64 byte accounting and overflow-safe giant-dimension rejection.
- [x] Require separate pairwise-output and item-gradient byte gates.
- [x] Prove an otherwise-valid public spatial estimator request fails before `_xi_nodes`.
- [x] Require exact Boolean `uses_space`, valid dimensions/itemsize/limit, and finite positive epsilon.
- [x] Pin one-matmul/in-place clamp/sqrt behavior and the absence of the 3-D broadcast.

Run:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py
```

Expected before production implementation: RED because the required helpers and constant do not exist and the legacy covariate broadcast remains.

---

## Task 2 — Checked byte products and preflight

**Files**

- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

- [ ] Add module-level `MAX_MARGINAL_WORKING_SET = 100_000_000` for the existing dominant element contract.
- [ ] Add `MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024` for float64 distance allocations.
- [ ] Implement `_checked_marginal_workspace_bytes(label, *dimensions, itemsize, limit_bytes=None)` using exact-integer validation and division-before-multiplication.
- [ ] Rework the existing dominant EM gate through an equivalent checked element helper; do not change its accepted dimensions.
- [ ] Implement `_validate_marginal_distance_workspaces` with exact-Boolean behavior, pairwise output bytes, and intentional `n_x × latent_dim` derivative bytes.
- [ ] Normalize model/rule/dimensions and invoke distance preflight before `_xi_nodes`.

Focused verification:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py -k 'workspace or public_estimator'
```

---

## Task 3 — In-place pairwise-distance helper

**Files**

- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

- [ ] Validate NumPy array type, rank, float64 dtype, shared positive latent width, finiteness, shapes, byte budget, and epsilon before BLAS.
- [ ] Compute one `left @ right.T` output.
- [ ] Mutate that output in place: multiply by `-2`, add row norms, clamp with `np.maximum(..., out=...)`, add epsilon, and apply `np.sqrt(..., out=...)`.
- [ ] Do not construct a second pairwise matrix or a 3-D difference.
- [ ] Preserve deterministic C-order output and finite checks.

Focused verification:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py -k 'pairwise'
```

---

## Task 4 — Rewire existing distance paths

**Files**

- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

- [ ] Reuse the helper in `_build_tables`.
- [ ] Reuse it for candidate item predictors.
- [ ] Reuse it for the tau M-step.
- [ ] Replace only the covariate `x_grid[None, :, :] - zeta[:, None, :]` broadcast.
- [ ] Retain the guarded `diff = x_grid - zeta_i[None, :]` required by the zeta derivative.
- [ ] Preserve non-spatial paths without distance allocation.

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
- Create: `benchmarks/benchmark_marginal_distance_workspaces.py`

- [ ] Add a moderate partially observed deterministic estimator case and compare all applicable returned finite arrays, likelihood trace, status, and iteration count across repeated identical runs.
- [ ] Exercise a valid covariate path and prove the helper is used without changing results.
- [ ] Record the largest existing repository fixtures accepted by the 128 MiB private ceiling.
- [ ] Add a safe command-line benchmark reporting Python, NumPy, BLAS, operating system, processor, dtype, dimensions, warm-ups, repetitions, elapsed distribution, and peak traced/RSS memory.
- [ ] Compare with the former broadcast only below a conservative benchmark safety limit.
- [ ] Describe all results as environment-specific; claim only removal of the named broadcast and enforcement of the byte ceiling.

---

## Task 6 — Doctoring and release-record parity

**Files**

- Create: `docs/doctoring/marginal-distance-workspace-bounds.md`
- Create: `docs/changelog.d/563-marginal-distance-workspace-bounds.md`
- Modify: `CHANGELOG.md`
- Modify: PR description

- [ ] Record the squared-norm equation, 128 MiB initial private budget rationale, in-place allocation model, precision treatment, compatibility evidence, failure modes, rollback, Rust ownership, and APA 7 references.
- [ ] Render the authoritative changelog with the repository renderer; do not hand-maintain a divergent block.
- [ ] Update the PR body from RED-only language to exact implemented behavior and evidence.

Commands:

```bash
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

---

## Task 7 — Exact-head verification and merge discipline

- [ ] Run focused tests first.
- [ ] Run complete Python statement/branch coverage and public-docstring gates.
- [ ] Run Rust workspace/all-target tests and clippy/fmt.
- [ ] Run PyO3, wheel reinstall, package acceptance, explicit GPU-no-skip, and fuzz.
- [ ] Require Security Scan and SAST on the same head.
- [ ] Request current-head CodeRabbit/OpenCode/Noema and a qualifying non-author approval.
- [ ] Require zero unresolved actionable threads.
- [ ] Remove `needs-revision` / `do-not-merge`, mark Ready, and enable auto-merge only after every gate is satisfied.
- [ ] Verify accepted `main`; close issue #563 only after protected merge and accepted-main evidence.
