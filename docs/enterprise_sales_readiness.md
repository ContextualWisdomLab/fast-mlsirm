# Acquisition and Enterprise Sales Readiness

## Position

This gate defines the evidence required before presenting `fast-mlsirm` for acquisition or enterprise procurement review. It is a product/procurement readiness standard, not a monetary target, valuation guarantee, customer claim, or regulated-use approval.

The current product is a local Python/Rust computation package for technical MLSIRM/IRT teams. Enterprise review must package the software with clear scope, exact-source acceptance evidence, support/security terms, privacy boundaries, and a customer validation plan.

## Canonical Buyer-Review Workflow

Run the price-neutral acquisition orchestrator on the exact candidate:

```bash
python scripts/build_acquisition_release.py \
  --out acquisition-release \
  --require-rust \
  --check-import
```

The default transaction scenario is unset. Supply `--contract-value-krw <value>` only when a real deal scenario needs to be recorded; that value is evidence metadata, not a quality threshold.

The orchestrator coordinates the existing package build/release-acceptance, benchmark, buyer-packet, release-index, procurement, PR-queue, and Figma evidence builders and finishes with `scripts/sales_readiness.py --require-acquisition-readiness` using every required artifact path explicitly. Missing evidence fails closed.

## Procurement Evidence

A candidate must provide evidence bound to one exact source/artifact set:

- Python tests, Rust tests, PyO3 tests, package build, metadata checks, and release acceptance succeed.
- `release_acceptance.py --require-rust` produces an `ok` acceptance summary.
- Built wheel and source distribution artifacts exist and are digest-bound.
- Installed package version matches project metadata and the Rust core imports when Rust support is offered.
- README, security, support, changelog, commercial-readiness, and release-acceptance guidance match the candidate.
- Benchmark JSON/HTML evidence records the observed release run and bounded runtime/scenario coverage.
- Buyer packet and release evidence index bind artifacts, source commit, acceptance, benchmark, and readiness evidence with SHA-256 digests.
- Procurement due diligence records package/policy/repository evidence without substituting documentation for actual customer, legal, or transfer authority.
- PR queue governance records current review/merge/staleness/release-scope evidence.
- Figma evidence sync confirms the bounded design packet and Code Connect-disabled policy where required.
- `final_acquisition_readiness_manifest.json` is `ok` with `require_acquisition_readiness=true` and `require_20b_product=false`.

## Customer Acceptance Evidence

A buyer-facing packet should include the exact commit SHA, package version, runtime/toolchain identity, backend used, distribution digests, acceptance/diagnostic reports, benchmark evidence, buyer/release index, procurement/queue/design evidence, and a synthetic-data reproduction path that does not expose customer response data.

These artifacts demonstrate the supplied evidence state only. Actual deployment, customer acceptance, contractual transfer, revenue, operational service levels, and legal authority remain independent evidence classes.

## Go / No-Go

A release is a `go` for enterprise review only when:

- all repository and applicable security/release checks for the unchanged candidate are terminal-clean;
- the price-neutral acquisition builder completes successfully;
- the final acquisition-readiness manifest has no failed checks and includes every required acquisition validator;
- no predecessor-head evidence is reused after source movement;
- no model formula, diagnostics semantics, or estimator scope changed outside the required scientific/model-design review;
- any live governance requirement such as independent approval is actually satisfied.

For an existing bundle, the canonical verifier can be re-run directly with `--require-acquisition-readiness` and the explicit benchmark, buyer-packet, release-index, procurement, PR-queue, and Figma manifest paths documented in [`release_acceptance.md`](release_acceptance.md).

## Out of Scope

The readiness gate does not claim clinical, educational-placement, hiring, or other regulated-decision suitability; hosted SaaS capabilities; customer adoption; valuation; revenue; universal fairness/validity; or performance guarantees beyond evidence generated for the candidate.

## Legacy 20B Compatibility

The deprecated compatibility profile retains the historical KRW 2,000,000,000 scenario and `--require-20b-product` for older automation. `scripts/build_commercial_release.py` remains the compatibility orchestrator on this revision. Neither is the current generic readiness path, and neither is evidence of value or quality. New automation uses `scripts/build_acquisition_release.py` / `--require-acquisition-readiness` and provides any transaction scenario explicitly.

## Operating Rule

Reviewer delay is not a source defect, but pending/queued evidence is not passing. A protected release can proceed only when source, tests, package artifacts, acquisition evidence, repository policy, and required independent approval are in a known current state. No temporary gate weakening, self-approval, force push, or routine administrator bypass substitutes for that state.
