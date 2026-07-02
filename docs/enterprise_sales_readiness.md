# Enterprise Sales Readiness

## Position

This gate defines the evidence required before presenting `fast-mlsirm` as a
KRW 2,000,000,000 enterprise sale candidate. It is a product and procurement
readiness standard, not a valuation guarantee and not a regulated-use approval.

The current product is a local Python/Rust computation package for technical
MLSIRM/IRT teams. A high-value enterprise sale should package the software with
clear scope, acceptance evidence, support terms, privacy boundaries, and a
customer validation plan.

For the KRW 2,000,000,000 product-readiness standard, this document is used with
`docs/20b_product_readiness.md`, `docs/buyer_demo_storyboard.md`,
`docs/figma_product_design_packet.md`, `docs/roi_evidence_model.md`, and
`examples/enterprise_demo/`.

## Procurement Evidence

A release candidate must provide the following evidence on the exact commit or
artifact being offered:

- Python tests, Rust tests, package build, wheel metadata checks, and release
  acceptance smoke all pass.
- `scripts/release_acceptance.py --require-rust` produces
  `acceptance_summary.json`.
- `scripts/sales_readiness.py` produces `sales_readiness_manifest.json` with
  every check marked `ok`.
- Built wheel and source distribution artifacts exist under `dist/`.
- Installed package version matches `pyproject.toml`.
- `fast_mlsirm._core.neg_loglik_and_grad` is importable from the installed
  artifact when Rust support is part of the offer.
- README, security policy, support policy, changelog, commercial readiness
  checklist, and release acceptance guide are present.
- Product Design storyboard, Figma design packet with Code Connect disabled,
  ROI model, benchmark manifest, and synthetic enterprise demo evidence are
  present when `--require-20b-product` is used.
- `scripts/build_benchmark_report.py` produces `benchmark_report.json` and
  `benchmark_report.html` from the exact release acceptance run.
- `scripts/build_release_evidence_index.py` produces
  `release_evidence_index.json` and `release_evidence_index.html` tying the
  dist artifacts, acceptance run, benchmark report, sales-readiness manifest,
  and buyer packet to one source commit.
- `scripts/build_commercial_release.py` produces
  `commercial_release_manifest.json` and `commercial_release_report.html` as a
  top-level buyer review summary over dist build, acceptance, benchmark,
  sales-readiness, buyer packet, release index, and final gate stages.

## Customer Acceptance Evidence

The buyer acceptance package should include:

- `acceptance_summary.json` from a clean release acceptance run.
- `sales_readiness_manifest.json` from the enterprise sales gate.
- Generated fit and dimensionality diagnostics JSON files.
- Generated standalone HTML reports for fit and dimensionality diagnostics.
- Generated benchmark JSON and HTML reports with runtime-budget and artifact
  coverage evidence.
- Generated release evidence JSON and HTML reports with distribution hashes,
  source commit, acceptance status, benchmark status, and buyer packet digest.
- Generated commercial release JSON and HTML reports with stage durations,
  failed-stage detail, artifact paths, and SHA256 digests.
- Exact commit SHA, package version, Python version, Rust toolchain version,
  operating system, and backend used.
- A synthetic-data reproduction path that does not expose customer response
  data.

## Go/No-Go

The release is a `go` for enterprise sales review only when all items below are
true:

- `python -m pytest` passes.
- `cargo test --workspace` passes.
- `python -m build` and `python -m twine check dist/*` pass.
- `python scripts/release_acceptance.py --out release-acceptance --require-rust`
  passes.
- `python scripts/build_benchmark_report.py --acceptance release-acceptance/acceptance_summary.json --out release-acceptance/benchmark`
  passes.
- `python scripts/sales_readiness.py --acceptance release-acceptance/acceptance_summary.json --dist dist --require-rust --check-import`
  passes.
- `python scripts/sales_readiness.py --acceptance release-acceptance/acceptance_summary.json --dist dist --require-rust --require-20b-product --benchmark-report release-acceptance/benchmark/benchmark_report.json --require-benchmark-report --check-import`
  passes for KRW 2,000,000,000 product-readiness review.
- `python scripts/build_buyer_packet.py --acceptance release-acceptance/acceptance_summary.json --sales-readiness release-acceptance/sales_readiness_manifest.json --dist dist --benchmark-report release-acceptance/benchmark/benchmark_report.json --out buyer-evidence-packet`
  passes when a portable procurement packet is part of the offer.
- `python scripts/build_release_evidence_index.py --acceptance release-acceptance/acceptance_summary.json --sales-readiness release-acceptance/sales_readiness_manifest.json --dist dist --benchmark-report release-acceptance/benchmark/benchmark_report.json --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json --out release-evidence-index`
  passes and emits `release_evidence_index.json` plus
  `release_evidence_index.html`.
- `python scripts/build_commercial_release.py --out commercial-release --require-rust --check-import`
  passes and emits `commercial_release_manifest.json` plus
  `commercial_release_report.html`.
- `python scripts/sales_readiness.py --acceptance release-acceptance/acceptance_summary.json --dist dist --require-rust --require-20b-product --benchmark-report release-acceptance/benchmark/benchmark_report.json --require-benchmark-report --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json --require-buyer-packet --release-evidence-index release-evidence-index/release_evidence_index.json --require-release-evidence-index --check-import`
  passes as the final evidence-integrity gate.
- No release candidate changes the formula contract, diagnostics semantics, or
  estimator scope outside a model-design PR.

## Out of Scope

The KRW 2,000,000,000 sales-readiness gate does not claim:

- clinical, educational-placement, hiring, or other regulated decision
  suitability;
- a hosted SaaS platform with authentication, billing, tenancy, or audit logs;
- Bayesian posterior inference;
- native GRM, GPCM, or GGUM ordinal estimators;
- sparse/block execution for very large response matrices;
- Figma Code Connect;
- a separately versioned library or submodule;
- performance guarantees beyond the release evidence generated for the
  specific candidate artifact.

## Operating Rule

Reviewer delay is not a code blocker. A release can proceed only when source,
tests, package artifacts, acceptance evidence, and repository policy are in a
known state. If an approval rule blocks an otherwise proven release and no
eligible reviewer exists, the merge operator may use a narrow, temporary policy
adjustment, merge the exact checked head, and restore the original rule
immediately.
