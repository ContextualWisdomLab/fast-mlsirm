# Bifactor Scoreability Indices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Rust-native scoreability diagnostics for fitted orthogonal bifactor models so users can decide whether a general score and residual domain scores are psychometrically defensible.

**Architecture:** A new `mlsirm_core::bifactor_indices` module accepts a standardized item-by-factor loading matrix, item uniquenesses, an explicit general-factor index, and a structural-zero tolerance. It computes ECV variants, item ECV, PUC when the loading pattern is a strict bifactor pattern, omega total, omega hierarchical, and construct replicability H. A separate explicitly named entry point converts logistic IRT slopes to a continuous latent-response loading solution. Integration tests pin the formulas to the published CRAN calculator example; documentation separates score interpretability from model selection and continuous latent-response omega from categorical observed-score reliability.

**Tech Stack:** Rust 2021, existing `mlsirm-core` crate, Cargo integration tests, repository documentation and CI.

## Global Constraints

- All numerical computation must execute in Rust; no Python numerical fallback is introduced.
- Every public item and non-obvious formula must have complete Rust documentation with APA 7th references.
- Production line and branch coverage remain 100% under the repository coverage contract.
- The primary API requires standardized orthogonal-factor loadings and explicit uniquenesses; it never silently converts raw IRT slopes.
- The optional logistic conversion is explicitly named `bifactor_latent_response_indices_from_logit_slopes`, states its `pi^2/3` residual convention, and cannot be described as categorical observed-score reliability.
- PUC is returned only for a strict bifactor loading pattern: every item loads on the general factor and on at most one specific factor.
- No universal scoreability cutoff is hard-coded; consumers must configure decision policies and report uncertainty.

---

### Task 1: Pin the numerical contract with integration tests

**Files:**
- Create: `crates/mlsirm-core/tests/bifactor_indices.rs`
- Create: `crates/mlsirm-core/tests/bifactor_logit_indices.rs`

**Interfaces:**
- Consumes: future `mlsirm_core::bifactor_indices::{bifactor_indices, bifactor_latent_response_indices_from_logit_slopes, BifactorIndicesConfig}`.
- Produces: executable formula, validation, strict-pattern, and logistic latent-response expectations for the Rust implementation.

- [ ] **Step 1: Write a failing published-example test**

Use the 12-item, four-factor loading matrix from the `BifactorIndicesCalculator` reference example and assert ECV, IECV, PUC, omega, omega hierarchical, H, factor item counts, and strict-pattern status to `1e-12`.

- [ ] **Step 2: Write failing structural and validation tests**

Cover non-strict cross-loadings (`puc == None`), zero-tolerance handling, invalid dimensions, out-of-range general factor, malformed lengths, empty items/factors, non-finite values, negative uniquenesses, invalid standardized loadings, and numerical underflow.

- [ ] **Step 3: Write the logistic latent-response RED contract**

Invert a known standardized loading solution to logistic slopes, recover the same latent-response indices, and reject malformed, non-finite, empty-row, and nondegenerate-limit inputs. Do not call the resulting omega coefficients categorical reliability.

- [ ] **Step 4: Commit the RED contract**

```bash
git add crates/mlsirm-core/tests/bifactor_indices.rs \
  crates/mlsirm-core/tests/bifactor_logit_indices.rs
git commit -m "test: pin bifactor scoreability index contract"
```

### Task 2: Implement the Rust scoreability kernel

**Files:**
- Create: `crates/mlsirm-core/src/bifactor_indices.rs`
- Modify: `crates/mlsirm-core/src/lib.rs`

**Interfaces:**
- Consumes: row-major standardized loadings and uniquenesses or explicitly supplied logistic slopes plus `BifactorIndicesConfig`.
- Produces: `BifactorIndicesResult` containing `ecv_ss`, `ecv_sg`, `ecv_gs`, `item_ecv`, optional `puc`, `omega_total`, `omega_hierarchical`, `construct_replicability`, factor counts, and strict-pattern status.

- [ ] **Step 1: Add validated public types**

Define `BifactorIndicesConfig { n_items, n_factors, general_factor, zero_tolerance }` and fully documented `BifactorIndicesResult`.

- [ ] **Step 2: Implement independent formula derivations**

Implement squared-loading sums, factor membership masks, strict bifactor-pattern detection, ECV variants, IECV, PUC, omega total, omega hierarchical, and H in one deterministic pass with checked size arithmetic.

- [ ] **Step 3: Implement numerically stable logistic latent-response conversion**

Use row-scaled coordinates for

`lambda_if = a_if / sqrt(sum_h a_ih^2 + pi^2/3)`

and

`psi_i = (pi^2/3) / (sum_h a_ih^2 + pi^2/3)`.

The public name and docstring must preserve the latent-response boundary.

- [ ] **Step 4: Register the module from the crate root**

Add `pub mod bifactor_indices;` to `crates/mlsirm-core/src/lib.rs`.

- [ ] **Step 5: Run focused tests**

```bash
cargo test -p mlsirm-core --test bifactor_indices
cargo test -p mlsirm-core --test bifactor_logit_indices
```

Expected: all tests pass.

- [ ] **Step 6: Commit the implementation**

```bash
git add crates/mlsirm-core/src/bifactor_indices.rs crates/mlsirm-core/src/lib.rs
git commit -m "feat: add Rust bifactor scoreability indices"
```

### Task 3: Add buyer-facing scientific documentation

**Files:**
- Create: `docs/bifactor_scoreability_indices.md`
- Modify: `docs/papers/implemented-literature-map.md`
- Create: `docs/changelog.d/401-bifactor-scoreability.md`

**Interfaces:**
- Consumes: the Task 2 API and formulas.
- Produces: interpretation boundaries, source governance, and discoverability for procurement and research review.

- [ ] **Step 1: Document formulas and semantics**

Explain each index, the strict-pattern requirement for PUC, the distinction between model selection and scoreability, why no universal cutoff is built into the kernel, and why logistic latent-response omega is not categorical observed-score reliability.

- [ ] **Step 2: Record source status in APA 7th form**

Mark the CRAN `BifactorIndicesCalculator` source files as read implementation oracles and Rodriguez, Reise, and Haviland (2016) as the cited methodological origin; avoid claiming the article was read in full.

- [ ] **Step 3: Update literature map and changelog fragment**

Add the Rust module and test paths to the implemented-literature map and create a release-aggregation fragment describing the diagnostics and scale boundary.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/bifactor_scoreability_indices.md \
  docs/papers/implemented-literature-map.md \
  docs/changelog.d/401-bifactor-scoreability.md
git commit -m "docs: document bifactor scoreability diagnostics"
```

### Task 4: Verify commercial-quality gates and open a draft PR

**Files:**
- Verify: all files changed in Tasks 1-3.

**Interfaces:**
- Consumes: completed implementation and documentation.
- Produces: reviewed pull request with reproducible evidence.

- [ ] **Step 1: Run Rust verification**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo llvm-cov --workspace --all-features --fail-under-lines 100
```

- [ ] **Step 2: Review the diff for formula and citation drift**

Confirm every output matches the independently calculated example values, every branch is tested, the docs do not treat scoreability indices as model-selection tests, and no binary/ordinal observed-score reliability claim is made without thresholds.

- [ ] **Step 3: Mark ready and enable auto-merge only after required checks pass**

```bash
gh pr ready
gh pr checks --watch
gh pr merge --auto --squash
```
