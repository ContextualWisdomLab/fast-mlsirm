# 20B Benchmark Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated benchmark evidence reporting and optional readiness validation for KRW 2,000,000,000 procurement review.

**Architecture:** Keep `fast-mlsirm` as one Python/Rust package. Add one standard-library report builder that consumes `acceptance_summary.json` and `benchmark_manifest.json`, then extend existing readiness and buyer-packet scripts to validate or include the generated report when requested.

**Tech Stack:** Python standard library (`argparse`, `json`, `hashlib`, `html.escape`, `pathlib`, `subprocess`), existing pytest tests, existing GitHub PR workflow.

## Global Constraints

- No new library, submodule, package split, hosted dashboard, frontend app, database, benchmark framework, or non-stdlib dependency.
- Figma Code Connect remains disabled.
- Do not change model formulas, estimators, diagnostics, or fit semantics.
- Preserve default readiness and buyer-packet behavior unless benchmark flags are supplied.

---

### Task 1: Benchmark Report Builder

**Files:**
- Create: `scripts/build_benchmark_report.py`
- Test: `tests/test_benchmark_report.py`

**Interfaces:**
- Consumes: `acceptance_summary.json` and `examples/enterprise_demo/benchmark_manifest.json`.
- Produces: `benchmark_report.json` and `benchmark_report.html`.

- [ ] **Step 1: Write report builder tests**

Create tests that assert `build_report(args)` records `status`, `budget_ok`, scenario coverage, required artifact coverage, `html_report_file`, and `html_report_sha256`.

- [ ] **Step 2: Implement report builder**

Parse acceptance steps, collect command durations, infer observed `auto` and `rust` backend coverage from fit steps, compare required artifact basenames, write static HTML with CSP and focusable tables, then write JSON.

- [ ] **Step 3: Verify builder**

Run: `python -m pytest tests/test_benchmark_report.py -q`
Expected: PASS.

### Task 2: Sales Readiness Benchmark Validation

**Files:**
- Modify: `scripts/sales_readiness.py`
- Modify: `tests/test_sales_readiness.py`

**Interfaces:**
- Consumes: `--benchmark-report` and `--require-benchmark-report`.
- Produces: `benchmark_report:*` checks in `sales_readiness_manifest.json`.

- [ ] **Step 1: Add readiness tests**

Add passing and SHA mismatch tests for a generated benchmark report.

- [ ] **Step 2: Implement optional validation**

Validate report JSON shape, `status == "ok"`, `budget_ok is True`, no missing scenario backends, no missing artifacts or paths, HTML existence, and matching HTML SHA256.

- [ ] **Step 3: Verify readiness tests**

Run: `python -m pytest tests/test_sales_readiness.py -q`
Expected: PASS.

### Task 3: Buyer Packet Inclusion

**Files:**
- Modify: `scripts/build_buyer_packet.py`
- Modify: `tests/test_buyer_evidence_packet.py`

**Interfaces:**
- Consumes: optional `--benchmark-report`.
- Produces: `benchmark/benchmark_report.json` and `benchmark/benchmark_report.html` inside the packet zip.

- [ ] **Step 1: Add packet inclusion test**

Assert the packet manifest marks `coverage.benchmark_report` true when the benchmark report is supplied and that both benchmark files are in the zip.

- [ ] **Step 2: Implement optional packet inclusion**

Read the benchmark report JSON, resolve `html_report_file`, include both files under `benchmark/`, and keep benchmark coverage optional for existing packet builds.

- [ ] **Step 3: Verify packet tests**

Run: `python -m pytest tests/test_buyer_evidence_packet.py -q`
Expected: PASS.

### Task 4: Product Evidence Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/20b_product_readiness.md`
- Modify: `docs/commercial_readiness.md`
- Modify: `docs/enterprise_sales_readiness.md`
- Modify: `docs/figma_product_design_packet.md`
- Modify: `docs/roi_evidence_model.md`
- Modify: `docs/buyer_demo_storyboard.md`
- Modify: `examples/enterprise_demo/README.md`
- Modify: `examples/enterprise_demo/benchmark_manifest.json`
- Modify: `examples/enterprise_demo/roi_manifest.json`
- Modify: `examples/enterprise_demo/figma_design_packet.json`
- Modify: `examples/enterprise_demo/product_completion_manifest.json`

**Interfaces:**
- Consumes: generated benchmark report contract.
- Produces: buyer-facing docs and machine-readable manifests that point to actual report outputs.

- [ ] **Step 1: Update docs and JSON manifests**

Document `build_benchmark_report.py`, `benchmark_report.json`, `benchmark_report.html`, `--benchmark-report`, and `--require-benchmark-report`.

- [ ] **Step 2: Validate JSON**

Run: `python -m json.tool examples/enterprise_demo/benchmark_manifest.json >/dev/null && python -m json.tool examples/enterprise_demo/roi_manifest.json >/dev/null && python -m json.tool examples/enterprise_demo/figma_design_packet.json >/dev/null && python -m json.tool examples/enterprise_demo/product_completion_manifest.json >/dev/null`
Expected: PASS.

### Task 5: Verification and PR Lifecycle

**Files:**
- No additional implementation files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: merged PR on `main`.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_benchmark_report.py tests/test_buyer_evidence_packet.py tests/test_sales_readiness.py -q`
Expected: PASS.

- [ ] **Step 2: Run full verification**

Run full Python tests, Rust tests, package build/twine, release acceptance, benchmark report build, benchmark-required sales readiness, buyer packet build with benchmark report, and buyer-packet-required sales readiness.

- [ ] **Step 3: PR lifecycle**

Push a `codex/` branch, open PR, address review comments, merge to `main`, and restore policy/ruleset state if temporary policy relaxation is needed.
