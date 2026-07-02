# 20B Buyer Evidence HTML Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> or equivalent stepwise execution. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Add a standalone buyer evidence HTML report and readiness validation
for KRW 2,000,000,000 procurement review.

**Architecture:** Keep `fast-mlsirm` as one Python/Rust package. Extend the
existing buyer packet builder with a standard-library HTML renderer and extend
the existing sales-readiness verifier with optional report validation.

**Tech Stack:** Python standard library (`html.escape`, `json`, `hashlib`,
`zipfile`, `pathlib`), existing pytest tests, existing GitHub PR workflow.

## Global Constraints

- No new library, submodule, hosted dashboard, frontend app, or JavaScript
  runtime.
- Figma Code Connect remains disabled.
- Do not change model formulas, estimators, diagnostics, or fit semantics.
- Preserve default readiness behavior unless packet validation flags are
  supplied.

---

### Task 1: HTML Report Generation

**Files:**
- Modify: `scripts/build_buyer_packet.py`
- Test: `tests/test_buyer_evidence_packet.py`

**Interfaces:**
- Consumes: the existing buyer evidence manifest data.
- Produces: `buyer_evidence_report.html` next to the manifest and packet zip.

- [ ] **Step 1: Add report assertions**

Assert that the packet builder creates the HTML report, records
`report_sha256`, includes a CSP meta tag, includes keyboard-focusable evidence
tables, shows packet ZIP SHA256, and stores the report in the packet zip.

- [ ] **Step 2: Implement standard-library renderer**

Use `html.escape` and static CSS only. Render coverage, contract value, source
commit, artifact count, packet ZIP SHA256, and file digest rows.

- [ ] **Step 3: Verify focused packet test**

Run: `python -m pytest tests/test_buyer_evidence_packet.py -q`
Expected: PASS.

### Task 2: Readiness Validation

**Files:**
- Modify: `scripts/sales_readiness.py`
- Modify: `tests/test_sales_readiness.py`
- Modify: `examples/enterprise_demo/product_completion_manifest.json`

**Interfaces:**
- Consumes: `report_file` and `report_sha256` from
  `buyer_evidence_manifest.json`.
- Produces: `buyer_packet:html_report` and
  `buyer_packet:html_report_sha256` checks.

- [ ] **Step 1: Extend buyer packet coverage**

Add `html_report` to required packet coverage when packet validation is
requested.

- [ ] **Step 2: Validate report presence and SHA256**

Resolve relative report paths from the manifest directory and compare the
recorded SHA256 to the local file.

- [ ] **Step 3: Verify focused readiness test**

Run: `python -m pytest tests/test_sales_readiness.py -q`
Expected: PASS.

### Task 3: Product Evidence Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/20b_product_readiness.md`
- Modify: `docs/commercial_readiness.md`
- Modify: `docs/buyer_demo_storyboard.md`
- Modify: `docs/figma_product_design_packet.md`
- Modify: `docs/roi_evidence_model.md`
- Modify: `examples/enterprise_demo/README.md`
- Create: `docs/superpowers/specs/2026-07-02-20b-buyer-evidence-html-review-design.md`
- Create: `docs/superpowers/plans/2026-07-02-20b-buyer-evidence-html-review.md`

**Interfaces:**
- Consumes: Figma `06-procurement-packet` design intent.
- Produces: repo documentation that points to the generated HTML report as the
  human-readable procurement review surface.

- [ ] **Step 1: Update docs and manifest**

Mention the HTML report in buyer-packet, Figma packet, ROI evidence, and
enterprise demo docs.

- [ ] **Step 2: Validate JSON**

Run: `python -m json.tool examples/enterprise_demo/product_completion_manifest.json >/dev/null`
Expected: PASS.

### Task 4: Verification and PR Lifecycle

**Files:**
- No additional implementation files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: merged PR on `main`.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_buyer_evidence_packet.py tests/test_sales_readiness.py -q`
Expected: PASS.

- [ ] **Step 2: Run full verification**

Run full Python tests, Rust tests, package build/twine, release acceptance,
sales readiness, packet build, and packet-required sales readiness.

- [ ] **Step 3: PR lifecycle**

Push a `codex/` branch, open PR, address review comments, merge to `main`, and
restore policy/ruleset state if temporary policy relaxation is needed.
