# Marginal Distance Workspace Bounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unbounded three-dimensional distance broadcast from the NumPy MMLE reference/fallback path, fail closed before every actual distance workspace allocation, and preserve numerical parity with the Rust-first estimator.

**Architecture:** Add one checked-product resource helper and one bounded pairwise Euclidean-distance helper in `python/fast_mlsirm/estimators/marginal.py`. Reuse the pairwise helper in table construction, tau updates, and covariate updates; retain only the guarded item-local gradient difference. Tests are written first and cover numerical equivalence, adversarial dimensions, source-level regression, and estimator behavior.

**Tech Stack:** Python 3.11+, NumPy, pytest, Rust/PyO3 production backend, GitHub Actions.

## Global Constraints

- Rust remains the production psychometric computation backend; Python remains a reference/fallback and orchestration path.
- The configured marginal resource ceiling is enforced before allocation with checked integer products.
- No public API, fit-result schema, model name, database object, or serialization identity changes.
- Added production statement and branch coverage is 100%; public callables have complete docstrings.
- Documentation uses APA 7th references and avoids universal performance claims.
- Exact-head CI, Rust/PyO3, package, explicit GPU-no-skip, fuzz, Security Scan, SAST, current-head review, independent approval, and zero unresolved threads are required before merge.

---

### Task 1: Add failing numerical and resource contracts

**Files:**
- Create: `tests/test_marginal_distance_workspace_bounds.py`
- Reference: `python/fast_mlsirm/estimators/marginal.py`

**Interfaces:**
- Consumes: current private marginal-estimator helpers.
- Produces: required private helper names `_checked_marginal_workspace_product`, `_validate_marginal_distance_workspaces`, and `_pairwise_euclidean_distances`.

- [ ] **Step 1: Write the failing helper tests**

```python
"""Resource and numerical contracts for marginal latent-space distances."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


def test_pairwise_distance_matches_explicit_broadcast() -> None:
    """The bounded identity must reproduce the former Euclidean equation."""
    left = np.array([[0.0, 0.0], [1.5, -0.5], [-2.0, 3.0]])
    right = np.array([[0.25, -0.75], [1.0, 2.0], [-2.0, 3.0], [4.0, -1.0]])
    expected = np.sqrt(1e-8 + np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2))

    actual = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-8,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_pairwise_distance_handles_zero_and_roundoff() -> None:
    """Identical and nearly identical rows remain finite and non-negative."""
    left = np.array([[1.0, 2.0, 3.0], [1e100, 1e100, 1e100]])
    right = left.copy()

    actual = marginal._pairwise_euclidean_distances(
        left,
        right,
        eps_distance=1e-12,
    )

    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(np.diag(actual), np.sqrt(1e-12), rtol=1e-12)


def test_distance_workspace_guard_fails_before_pairwise_allocation(monkeypatch) -> None:
    """Oversized output dimensions fail without asking NumPy for the matrix."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_WORKING_SET", 1_000)

    with pytest.raises(ValueError, match="pairwise distance workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=101,
            n_x=10,
            latent_dim=3,
            uses_space=True,
        )


def test_distance_workspace_guard_fails_before_gradient_allocation(monkeypatch) -> None:
    """The item-local node-by-dimension gradient workspace is also bounded."""
    monkeypatch.setattr(marginal, "MAX_MARGINAL_WORKING_SET", 1_000)

    with pytest.raises(ValueError, match="item distance gradient workspace"):
        marginal._validate_marginal_distance_workspaces(
            n_items=2,
            n_x=501,
            latent_dim=2,
            uses_space=True,
        )


@pytest.mark.parametrize(
    "dimensions",
    [(-1, 2), (True, 2), (2.5, 2), (2, False)],
)
def test_checked_workspace_product_rejects_invalid_dimensions(dimensions) -> None:
    """Malformed dimensions never reach multiplication or allocation."""
    with pytest.raises(ValueError, match="non-negative integers"):
        marginal._checked_marginal_workspace_product("test workspace", *dimensions)


def test_non_spatial_models_skip_distance_workspace_contract() -> None:
    """MIRT does not incur latent-space distance workspaces."""
    marginal._validate_marginal_distance_workspaces(
        n_items=10**9,
        n_x=10**9,
        latent_dim=3,
        uses_space=False,
    )


def test_covariate_path_has_no_three_dimensional_distance_broadcast() -> None:
    """The removed item-by-node-by-dimension temporary cannot silently return."""
    source = inspect.getsource(marginal.fit_marginal_numpy)
    assert "x_grid[None, :, :] - zeta[:, None, :]" not in source
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py
```

Expected: failures because the three private helpers do not yet exist and the legacy three-dimensional covariate broadcast is still present.

- [ ] **Step 3: Commit the RED contract**

```bash
git add tests/test_marginal_distance_workspace_bounds.py
git commit -m "test(mmle): define distance workspace bounds"
```

---

### Task 2: Implement checked resource products and pairwise distances

**Files:**
- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

**Interfaces:**
- Consumes: module-level `MAX_MARGINAL_WORKING_SET`, finite float64 matrices.
- Produces: the three private helpers specified in Task 1.

- [ ] **Step 1: Promote the resource ceiling to module scope**

Add beside the existing module constants:

```python
MAX_MARGINAL_WORKING_SET = 100_000_000
```

Remove the function-local constant so tests and every helper share one authoritative limit.

- [ ] **Step 2: Add a checked dimension-product helper**

```python
def _checked_marginal_workspace_product(
    label: str,
    *dimensions: int,
    limit: int | None = None,
) -> int:
    """Return a bounded product for one marginal-estimator workspace.

    Boolean, non-integral, and negative dimensions fail before multiplication.
    The division-form limit check prevents an oversized product from being used
    as an allocation shape even on runtimes with arbitrary-precision integers.
    """
    ceiling = MAX_MARGINAL_WORKING_SET if limit is None else limit
    if (
        not isinstance(ceiling, (int, np.integer))
        or isinstance(ceiling, (bool, np.bool_))
        or int(ceiling) < 0
    ):
        raise ValueError("workspace limit must be a non-negative integer")
    product = 1
    for dimension in dimensions:
        if (
            not isinstance(dimension, (int, np.integer))
            or isinstance(dimension, (bool, np.bool_))
            or int(dimension) < 0
        ):
            raise ValueError(f"{label} dimensions must be non-negative integers")
        value = int(dimension)
        if value and product > int(ceiling) // value:
            raise ValueError(
                f"{label} exceeds the {int(ceiling)}-element limit"
            )
        product *= value
    return product
```

- [ ] **Step 3: Add the distance-workspace validator**

```python
def _validate_marginal_distance_workspaces(
    *,
    n_items: int,
    n_x: int,
    latent_dim: int,
    uses_space: bool,
) -> None:
    """Fail before allocating any active latent-space distance workspace."""
    if not isinstance(uses_space, (bool, np.bool_)):
        raise ValueError("uses_space must be boolean")
    if not uses_space:
        return
    _checked_marginal_workspace_product(
        "item position workspace", n_items, latent_dim
    )
    _checked_marginal_workspace_product(
        "latent node workspace", n_x, latent_dim
    )
    _checked_marginal_workspace_product(
        "item distance gradient workspace", n_x, latent_dim
    )
    _checked_marginal_workspace_product(
        "pairwise distance workspace", n_items, n_x
    )
```

- [ ] **Step 4: Add the bounded pairwise-distance helper**

```python
def _pairwise_euclidean_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    eps_distance: float,
) -> np.ndarray:
    """Return row-wise pairwise distances without a 3-D broadcast workspace."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("pairwise distance inputs must be two-dimensional")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("pairwise distance inputs must share one latent dimension")
    if not np.isfinite(eps_distance) or eps_distance <= 0.0:
        raise ValueError("eps_distance must be finite and positive")
    _validate_marginal_distance_workspaces(
        n_items=left_array.shape[0],
        n_x=right_array.shape[0],
        latent_dim=left_array.shape[1],
        uses_space=True,
    )
    left_sq = np.einsum("ij,ij->i", left_array, left_array)
    right_sq = np.einsum("ij,ij->i", right_array, right_array)
    squared = left_sq[:, None] + right_sq[None, :] - 2.0 * (left_array @ right_array.T)
    return np.sqrt(eps_distance + np.maximum(squared, 0.0))
```

- [ ] **Step 5: Run helper tests and verify GREEN**

Run:

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py
```

Expected: helper tests pass except the source-level no-3D-broadcast test, which remains RED until Task 3.

- [ ] **Step 6: Commit the helper implementation**

```bash
git add python/fast_mlsirm/estimators/marginal.py tests/test_marginal_distance_workspace_bounds.py
git commit -m "fix(mmle): bound distance workspaces"
```

---

### Task 3: Replace every avoidable distance broadcast

**Files:**
- Modify: `python/fast_mlsirm/estimators/marginal.py`
- Test: `tests/test_marginal_distance_workspace_bounds.py`

**Interfaces:**
- Consumes: `_pairwise_euclidean_distances` and `_validate_marginal_distance_workspaces`.
- Produces: no three-dimensional item-node-dimension distance temporary.

- [ ] **Step 1: Validate dimensions before latent-node construction**

Move model normalization and `_model_flags(model)` before node creation, derive the exact `_nx`, and call:

```python
_validate_marginal_distance_workspaces(
    n_items=n_items,
    n_x=_nx,
    latent_dim=latent_dim,
    uses_space=uses_space,
)
```

Keep the existing dominant EM working-set check and use `_checked_marginal_workspace_product` rather than an unchecked multiplication.

- [ ] **Step 2: Use the pairwise helper in `_build_tables`**

Replace the repeated squared-norm block with:

```python
dist = _pairwise_euclidean_distances(
    zeta,
    x_grid,
    eps_distance=eps_distance,
)
```

- [ ] **Step 3: Reuse the helper in item and tau updates**

For candidate item parameters, call the helper with `zeta_c[None, :]` and select row zero. For the zeta gradient, retain `diff = x_grid - zeta_i[None, :]`; its exact `n_x × latent_dim` allocation was validated before the loop.

For the tau update, replace the local pairwise formula with the shared helper.

- [ ] **Step 4: Remove the covariate three-dimensional broadcast**

Replace:

```python
diffz = x_grid[None, :, :] - zeta[:, None, :]
distz = np.sqrt(eps_distance + np.sum(diffz * diffz, axis=2))
```

with:

```python
distz = _pairwise_euclidean_distances(
    zeta,
    x_grid,
    eps_distance=eps_distance,
)
```

- [ ] **Step 5: Run focused contracts**

```bash
pytest -q tests/test_marginal_distance_workspace_bounds.py
```

Expected: all tests pass, including the source-level regression.

- [ ] **Step 6: Commit the production rewiring**

```bash
git add python/fast_mlsirm/estimators/marginal.py tests/test_marginal_distance_workspace_bounds.py
git commit -m "perf(mmle): avoid 3d distance broadcasts"
```

---

### Task 4: Add realistic estimator parity and allocation evidence

**Files:**
- Modify: `tests/test_marginal_distance_workspace_bounds.py`
- Create: `benchmarks/benchmark_marginal_distance_workspaces.py`

**Interfaces:**
- Consumes: public/fallback estimator and private pairwise helper.
- Produces: deterministic parity evidence and reproducible peak-allocation benchmark output.

- [ ] **Step 1: Add a realistic partially observed regression**

Use a small connected binary response matrix, `model="MLS2PLM"`, two latent dimensions, QMC nodes, and one iteration. Independently compute the former broadcast distances for the initialized item coordinates and assert the helper matches exactly within `rtol=1e-12, atol=1e-12`. Run the estimator twice with identical seed and assert all returned finite arrays, status, and likelihood trace are deterministic.

- [ ] **Step 2: Add a standalone benchmark**

The benchmark must accept command-line dimensions, use `time.perf_counter_ns`, `tracemalloc`, and repeated warm-up/measured runs, and print JSON containing:

```json
{
  "numpy_version": "...",
  "blas": "...",
  "dtype": "float64",
  "n_items": 1000,
  "n_x": 256,
  "latent_dim": 3,
  "warmups": 3,
  "repetitions": 10,
  "elapsed_ns": [],
  "python_peak_bytes": 0
}
```

It must compare the bounded helper against the former explicit broadcast only for dimensions that fit a conservative benchmark safety limit. It must label results as environment-specific.

- [ ] **Step 3: Run focused and existing marginal tests**

```bash
pytest -q \
  tests/test_marginal_distance_workspace_bounds.py \
  tests/test_estimator_marginal.py \
  tests/test_objective.py
```

Expected: PASS.

- [ ] **Step 4: Commit parity and benchmark evidence**

```bash
git add tests/test_marginal_distance_workspace_bounds.py benchmarks/benchmark_marginal_distance_workspaces.py
git commit -m "test(mmle): verify bounded distance parity"
```

---

### Task 5: Document the decision and render release notes

**Files:**
- Create: `docs/doctoring/marginal-distance-workspace-bounds.md`
- Create: `docs/changelog.d/563-marginal-distance-workspaces.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: implemented behavior and measured evidence.
- Produces: APA 7th equation-to-code traceability, rollback, and buyer-readable release note.

- [ ] **Step 1: Write the doctoring record**

Document:

- the former three-dimensional broadcast;
- the squared-norm identity;
- exact workspace limits and error precedence;
- floating-point clamp rationale;
- Rust/Python ownership boundary;
- benchmark environment and non-universality statement;
- verification commands and rollback;
- APA 7th references from the design.

- [ ] **Step 2: Add the authoritative changelog fragment**

State that the NumPy fallback now rejects oversized latent-distance workspaces before allocation and avoids the item-node-dimension broadcast while preserving the Rust-first numerical contract. Do not claim a universal speedup.

- [ ] **Step 3: Run the repository changelog renderer/check**

Use the repository's existing managed-fragment command and verify `CHANGELOG.md` exactly reflects the fragment.

- [ ] **Step 4: Commit documentation and changelog**

```bash
git add docs/doctoring/marginal-distance-workspace-bounds.md \
  docs/changelog.d/563-marginal-distance-workspaces.md CHANGELOG.md
git commit -m "docs: record marginal distance workspace bounds"
```

---

### Task 6: Complete exact-head verification and merge gates

**Files:**
- No new production files unless a verified failure requires a minimal fix.

**Interfaces:**
- Consumes: exact current head.
- Produces: mergeable, independently approved PR with no unresolved finding.

- [ ] **Step 1: Run local/full verification**

```bash
pytest -q
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

Run the repository's package reinstall, explicit GPU-no-skip, fuzz, coverage, docstring, and release-acceptance commands as defined by current workflows.

- [ ] **Step 2: Verify added-code coverage**

Require 100% statement and branch coverage for modified production paths. Add focused tests only for uncovered reachable branches; do not use exclusions to hide behavior.

- [ ] **Step 3: Request exact-head review**

Request CodeRabbit, OpenCode, Noema, and required human/team reviewers on the exact unchanged head. Resolve valid findings with new RED→GREEN regressions and restart exact-head verification after every commit.

- [ ] **Step 4: Enable protected auto-merge**

Enable auto-merge only when all required checks are successful on the exact head, the PR is not Draft, every actionable thread is resolved, and a qualifying non-author approval exists.

- [ ] **Step 5: Verify protected-main result**

After merge, verify `main` CI and the next hourly governance snapshot. Close issue #563 only when the accepted commit and evidence are linked.

## Plan self-review

- Spec coverage: every issue #563 requirement maps to Tasks 1–6.
- Placeholder scan: no TBD, TODO, or unspecified implementation step remains.
- Type consistency: helper names and signatures are consistent across tests, implementation, documentation, and plan.
- Scope: one bounded NumPy reference/fallback reliability slice; no unrelated estimator refactor or new production backend.
