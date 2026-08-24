# Buyer Demo Storyboard

## Audience

The demo is for a technical buyer evaluating whether `fast-mlsirm` can become a
procurement-backed local analytics package for MLSIRM/IRT model workflows.

## Narrative

The demo should show a complete buyer review without customer data:

1. **Package Evidence**
   - Show version, wheel, source distribution, Python version, Rust toolchain,
     and commit SHA.
   - Show `commercial_release_manifest.json` as the top-level evidence index
     for the buyer run.
   - Explain that NumPy is the reference backend and Rust/PyO3 is optional
     acceleration for the objective.
2. **Synthetic Data**
   - Generate MLS2PLM synthetic response data.
   - Show that no buyer data is required for acceptance reproduction.
   - Open `benchmark_report.html` to review runtime budget, command duration,
     backend coverage, and required artifact coverage.
3. **Fit Workflow**
   - Run fitting with `backend=auto` and explicit Rust evidence when required.
   - Show `fit_summary.json`, resolved backend, objective, log-likelihood, and
     generated parameter artifact.
4. **Diagnostics Workflow**
   - Show fit diagnostics, dimensionality diagnostics, and response-process
     diagnostics.
   - Keep the interpretation bounded to point-estimate diagnostics.
5. **Report Review**
   - Open the standalone HTML report.
   - Show item/person/model fit summaries, dimensionality candidate comparison,
     and report coverage notes.
6. **Procurement Packet**
   - Show `acceptance_summary.json`, `sales_readiness_manifest.json`,
     `roi_manifest.json`, `benchmark_manifest.json`, and
     `figma_design_packet.json`.
   - Show `buyer_evidence_manifest.json` and
     `fast_mlsirm_buyer_evidence_packet.zip` when the buyer wants a single
     portable evidence bundle.
   - Open `buyer_evidence_report.html` to review coverage, contract value,
     source commit, artifact count, and digest evidence without reading raw
     JSON first.
   - Show benchmark report JSON and HTML files when they are included in the
     packet.
   - Open `release_evidence_index.html` to review wheel/source distribution
     hashes, acceptance status, benchmark status, sales-readiness status, buyer
     packet digest, source commit, and required evidence coverage in one place.
   - Open `commercial_release_report.html` to review the full stage timeline,
     failed-stage detail when present, artifact paths, and SHA256 evidence from
     the single command.
   - Open `procurement_due_diligence_report.html` to review package metadata,
     policy-file checks, commercial-release integrity, GitHub snapshot state,
     failed-check detail, and report SHA256 evidence.
   - Open `pr_queue_governance_report.html` to review open PR review state,
     stale and changes-requested counts, release-scope conflict classification,
     and report SHA256 evidence.
   - Open `figma_evidence_sync_report.html` to review Code Connect-disabled
     design packet coverage, required procurement evidence tokens, optional
     Figma metadata snapshot status, and report SHA256 evidence.
7. **IRT Stability Review**
   - Show missing-by-design response handling, true-parameter reproduction,
     Hessian/vcov/standard-error evidence, second-order status, fixed item
     linking, CAT item selection, and ATA form assembly support.
   - Open `docs/irt_stability_product_design.md` as the source for the
     Information Architecture, screen definition, key screen, wireframe, and
     user stories.

## Screen List For Figma

The Figma prototype should contain these static screens:

- `01-package-evidence`: artifact and environment checklist.
- `02-synthetic-demo-run`: CLI/API workflow with generated files and benchmark
  report evidence.
- `03-fit-diagnostics`: itemfit, personfit, model fit, backend, and runtime
  cards.
- `04-dimensionality-review`: candidate dimensions and selected model summary.
- `05-report-export`: standalone HTML report review state.
- `06-procurement-packet`: required manifests, artifact digests, packet zip,
  standalone HTML review, benchmark report files, release evidence index, and
  commercial release report with procurement due-diligence and PR queue
  governance go/no-go status plus Figma evidence sync status.
- `07-irt-stability-review`: missingness, true-parameter, Hessian/vcov/SE,
  fixed-item linking, CAT, and ATA stability evidence.

## Interaction Level

The first product-design pass is static. Static is intentional: the shipped
program is currently a local package and CLI, not a hosted dashboard. A fully
interactive prototype becomes useful only after a buyer asks for a hosted
review surface.

## Acceptance

The storyboard is complete when every screen maps to a real CLI/API artifact
or a manifest checked by `scripts/sales_readiness.py --require-20b-product`.
The procurement-packet screen also maps to `scripts/build_buyer_packet.py` and
optional `scripts/sales_readiness.py --require-buyer-packet` validation, plus
`scripts/build_release_evidence_index.py` and optional
`scripts/sales_readiness.py --require-release-evidence-index` validation. The
top-level buyer run maps to `scripts/build_commercial_release.py`, which writes
`commercial_release_manifest.json` and `commercial_release_report.html`, then
runs `scripts/build_procurement_due_diligence.py` to write
`procurement_due_diligence_manifest.json` and
`procurement_due_diligence_report.html`, and runs
`scripts/build_pr_queue_governance.py` to write
`pr_queue_governance_manifest.json` and `pr_queue_governance_report.html`.
It then runs `scripts/build_figma_evidence_sync.py` to write
`figma_evidence_sync_manifest.json` and `figma_evidence_sync_report.html`.
The IRT stability review screen maps to
`docs/irt_stability_product_design.md` and
`tests/test_irt_stability.py`.
The final optional integrity checks are
`scripts/sales_readiness.py --require-procurement-due-diligence` and
`scripts/sales_readiness.py --require-pr-queue-governance`, plus
`scripts/sales_readiness.py --require-figma-evidence-sync` when the design
packet is part of the buyer review.
