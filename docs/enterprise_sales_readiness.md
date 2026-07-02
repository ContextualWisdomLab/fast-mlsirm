# Enterprise Sales Readiness

## Position

This gate defines the evidence required before presenting `fast-mlsirm` as a
KRW 2,000,000,000 enterprise sale candidate. It is a product and procurement
readiness standard, not a valuation guarantee and not a regulated-use approval.

The current product is a local Python/Rust computation package for technical
MLSIRM/IRT teams. A high-value enterprise sale should package the software with
clear scope, acceptance evidence, support terms, privacy boundaries, and a
customer validation plan.

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

## Customer Acceptance Evidence

The buyer acceptance package should include:

- `acceptance_summary.json` from a clean release acceptance run.
- `sales_readiness_manifest.json` from the enterprise sales gate.
- Generated fit and dimensionality diagnostics JSON files.
- Generated standalone HTML reports for fit and dimensionality diagnostics.
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
- `python scripts/sales_readiness.py --acceptance release-acceptance/acceptance_summary.json --dist dist --require-rust --check-import`
  passes.
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
- performance guarantees beyond the release evidence generated for the
  specific candidate artifact.

## Operating Rule

Reviewer delay is not a code blocker. A release can proceed only when source,
tests, package artifacts, acceptance evidence, and repository policy are in a
known state. If an approval rule blocks an otherwise proven release and no
eligible reviewer exists, the merge operator may use a narrow, temporary policy
adjustment, merge the exact checked head, and restore the original rule
immediately.
