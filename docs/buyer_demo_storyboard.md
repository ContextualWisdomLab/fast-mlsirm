# Buyer Demo Storyboard

## Audience

The demo is for a technical buyer evaluating whether `fast-mlsirm` can become a
procurement-backed local analytics package for MLSIRM/IRT model workflows.

## Narrative

The demo should show a complete buyer review without customer data:

1. **Package Evidence**
   - Show version, wheel, source distribution, Python version, Rust toolchain,
     and commit SHA.
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
  go/no-go status.

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
`scripts/sales_readiness.py --require-release-evidence-index` validation.
