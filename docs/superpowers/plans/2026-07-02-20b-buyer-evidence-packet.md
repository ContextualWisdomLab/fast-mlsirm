# 20B Buyer Evidence Packet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable buyer evidence packet and optional readiness validation for KRW 2,000,000,000 procurement review.

**Architecture:** Keep the current Python/Rust package as the sellable unit. Add one standard-library script that gathers existing release and product evidence into a zip plus JSON manifest, then extend the existing sales-readiness verifier to validate that manifest when requested.

**Tech Stack:** Python standard library (`argparse`, `json`, `hashlib`, `zipfile`, `pathlib`, `subprocess`), existing pytest tests, existing GitHub PR workflow.

## Global Constraints

- No new library, submodule, hosted SaaS, artifact registry, or signing service.
- Figma Code Connect remains disabled.
- Do not alter formulas, estimators, or diagnostics semantics.
- Preserve default `sales_readiness.py` behavior unless packet validation flags are supplied.

---

### Task 1: Packet Builder

**Files:**
- Create: `scripts/build_buyer_packet.py`
- Test: `tests/test_buyer_evidence_packet.py`

**Interfaces:**
- Consumes: `acceptance_summary.json`, `sales_readiness_manifest.json`, `dist/*.whl`, `dist/*.tar.gz`, product docs and demo manifests.
- Produces: `buyer_evidence_manifest.json` and `fast_mlsirm_buyer_evidence_packet.zip`.

- [ ] **Step 1: Write packet builder tests**

Run: `python -m pytest tests/test_buyer_evidence_packet.py -q`
Expected: FAIL before the script exists.

- [ ] **Step 2: Implement packet builder**

Use `zipfile.ZipFile`, `hashlib.sha256`, and `json`. Fail when required
coverage is missing.

- [ ] **Step 3: Verify packet builder tests**

Run: `python -m pytest tests/test_buyer_evidence_packet.py -q`
Expected: PASS.

### Task 2: Sales Readiness Validation

**Files:**
- Modify: `scripts/sales_readiness.py`
- Modify: `tests/test_sales_readiness.py`

**Interfaces:**
- Consumes: `--buyer-packet-manifest` and `--require-buyer-packet`.
- Produces: `buyer_packet:*` checks inside `sales_readiness_manifest.json`.

- [ ] **Step 1: Add failing readiness tests**

Run: `python -m pytest tests/test_sales_readiness.py -q`
Expected: FAIL until packet validation is implemented.

- [ ] **Step 2: Implement optional packet validation**

Check status, contract value, artifact count, coverage booleans, zip existence,
and zip SHA256.

- [ ] **Step 3: Verify readiness tests**

Run: `python -m pytest tests/test_sales_readiness.py -q`
Expected: PASS.

### Task 3: Product Evidence Docs

**Files:**
- Modify: `docs/20b_product_readiness.md`
- Modify: `docs/commercial_readiness.md`
- Modify: `docs/buyer_demo_storyboard.md`
- Modify: `docs/figma_product_design_packet.md`
- Modify: `docs/roi_evidence_model.md`
- Modify: `examples/enterprise_demo/README.md`
- Modify: `examples/enterprise_demo/product_completion_manifest.json`

**Interfaces:**
- Consumes: packet builder command and readiness packet checks.
- Produces: buyer-facing explanation that maps Figma procurement packet to real artifacts.

- [ ] **Step 1: Update docs and manifest**

Describe `buyer_evidence_manifest.json`, packet zip, SHA256 coverage, and the
optional `--require-buyer-packet` gate.

- [ ] **Step 2: Validate JSON**

Run: `python -m json.tool examples/enterprise_demo/product_completion_manifest.json >/dev/null`
Expected: PASS.

### Task 4: Verification and PR

**Files:**
- No new implementation files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: merged PR.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_buyer_evidence_packet.py tests/test_sales_readiness.py -q`
Expected: PASS.

- [ ] **Step 2: Run full verification**

Run Python tests, Rust tests, package build/twine, release acceptance, buyer
packet build, and 20B sales readiness with packet validation.

- [ ] **Step 3: PR lifecycle**

Push a `codex/` branch, open PR, address review comments, merge to `main`, and
restore rulesets if policy was temporarily relaxed.
