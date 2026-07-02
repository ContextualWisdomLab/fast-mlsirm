# KRW 2,000,000,000 Product Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add buyer-facing product, design, analytics, and automated evidence gates for KRW 2,000,000,000 enterprise procurement review.

**Architecture:** Reuse the existing package and readiness script. Add static docs and JSON manifests for Product Design, Figma, Data Analytics, ROI, and benchmark evidence, then validate them through `scripts/sales_readiness.py --require-20b-product`.

**Tech Stack:** Python standard library, pytest, GitHub Actions, Markdown, JSON, existing Python/Rust package.

## Global Constraints

- No MLSIRM formula, estimator, simulation, or diagnostics semantic changes.
- No Figma Code Connect.
- No new library, submodule, hosted SaaS, auth, billing, tenancy, or audit-log layer.
- Synthetic demo evidence only.
- Review delay is not a blocker after source, tests, package, CI, and policy state are known.

---

### Task 1: Product Evidence Artifacts

**Files:**
- Create: `docs/20b_product_readiness.md`
- Create: `docs/buyer_demo_storyboard.md`
- Create: `docs/figma_product_design_packet.md`
- Create: `docs/roi_evidence_model.md`
- Create: `examples/enterprise_demo/README.md`
- Create: `examples/enterprise_demo/roi_manifest.json`
- Create: `examples/enterprise_demo/benchmark_manifest.json`
- Create: `examples/enterprise_demo/figma_design_packet.json`

**Interfaces:**
- Produces: required artifacts consumed by `scripts/sales_readiness.py --require-20b-product`.

- [ ] **Step 1: Add product, design, Figma, ROI, and demo evidence files**
- [ ] **Step 2: Check each JSON file parses**

Run:

```bash
python -m json.tool examples/enterprise_demo/roi_manifest.json >/dev/null
python -m json.tool examples/enterprise_demo/benchmark_manifest.json >/dev/null
python -m json.tool examples/enterprise_demo/figma_design_packet.json >/dev/null
```

Expected: all commands exit 0.

### Task 2: Readiness Script Validation

**Files:**
- Modify: `scripts/sales_readiness.py`
- Test: `tests/test_sales_readiness.py`

**Interfaces:**
- Consumes: product evidence artifacts from Task 1.
- Produces: `--require-20b-product` CLI flag and 20B product checks in `sales_readiness_manifest.json`.

- [ ] **Step 1: Add tests for the new product gate**
- [ ] **Step 2: Run the tests and observe failure**

Run:

```bash
python -m pytest tests/test_sales_readiness.py -q
```

Expected before implementation: failure for missing product gate.

- [ ] **Step 3: Implement the product gate with standard-library JSON checks**
- [ ] **Step 4: Re-run the focused tests**

Run:

```bash
python -m pytest tests/test_sales_readiness.py -q
```

Expected: pass.

### Task 3: Documentation And CI Wiring

**Files:**
- Modify: `README.md`
- Modify: `docs/commercial_readiness.md`
- Modify: `docs/enterprise_sales_readiness.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `--require-20b-product` from Task 2.
- Produces: user-facing instructions and CI enforcement.

- [ ] **Step 1: Document the new gate**
- [ ] **Step 2: Add the flag to CI package job**
- [ ] **Step 3: Run local gate command after release acceptance exists**

Run:

```bash
python scripts/sales_readiness.py --acceptance release-acceptance/acceptance_summary.json --dist dist --require-rust --require-20b-product --check-import --out release-acceptance/sales_readiness_manifest.json
```

Expected: JSON status is `ok`.

### Task 4: Full Verification And PR

**Files:**
- All files from Tasks 1-3.

**Interfaces:**
- Produces: PR ready for review and merge.

- [ ] **Step 1: Run Python tests**
- [ ] **Step 2: Run Rust tests**
- [ ] **Step 3: Build package and twine check**
- [ ] **Step 4: Run release acceptance and sales readiness with 20B product gate**
- [ ] **Step 5: Commit, push, open PR, handle review comments, and merge**
