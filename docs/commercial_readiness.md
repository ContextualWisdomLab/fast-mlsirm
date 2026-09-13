# Commercial Readiness

## Readiness Position

`fast-mlsirm` is ready for commercial beta deployment by technical users who understand MLSIRM/IRT workflows and can evaluate model suitability for their own domain. It is not positioned as a finished regulated decision product or as a fully managed assessment platform.

For acquisition or commercial review, the current gate is **evidence completeness**, not a monetary target or valuation claim. Use [`scripts/build_acquisition_release.py`](../scripts/build_acquisition_release.py) for the complete buyer-review bundle and [`scripts/sales_readiness.py`](../scripts/sales_readiness.py) with `--require-acquisition-readiness` for the canonical final decision.

## Supported Product Surface

- Python API for simulation, fitting, diagnostics, recovery checks, and report rendering.
- CLI workflows for simulation, fitting, fit diagnostics, dimensionality diagnostics, response-process diagnostics, and report rendering.
- Rust/PyO3 backend as the default `auto` runtime path through `fast_mlsirm._core`; `auto` fails closed when the compiled core is unavailable.
- Explicit NumPy reference backend for parity testing only.
- Dense response matrices with missing values represented by `NaN`, `-1`, or an explicit mask.
- Automated benchmark, buyer-packet, release-index, procurement, PR-queue, and Figma evidence reporting.
- A price-neutral single-command acquisition evidence builder through `scripts/build_acquisition_release.py`.

## Not Yet Supported

- Sparse/block execution for very large matrices.
- Posterior predictive checking and Bayesian posterior inference.
- Native ordinal response-model estimation such as GRM, GPCM, or GGUM.
- Hosted dashboards, user management, billing, or enterprise administration.
- Domain-specific clinical, employment, or educational placement decisions.

## Seller Acceptance Checklist

The **Enterprise Sales Gate** is the same evidence-bound acquisition-readiness decision; it is not a separate monetary threshold.

Before treating a build as ready for buyer review, verify the exact release commit with the repository-owned test/package gates, then run:

```bash
python scripts/build_acquisition_release.py \
  --out acquisition-release \
  --require-rust \
  --check-import
```

The generic command leaves transaction value unset. If a real deal scenario must be recorded, add `--contract-value-krw <value>` explicitly; that value remains transaction metadata rather than a product-quality threshold.

The complete run must produce an `ok` `release-acceptance/final_acquisition_readiness_manifest.json` and evidence for the exact source candidate covering:

- release acceptance and installed Rust-core import evidence;
- wheel and source distribution artifacts;
- benchmark evidence;
- buyer packet and release evidence index with SHA-256 digests;
- procurement due diligence;
- PR queue governance;
- Figma evidence synchronization;
- current README, PRD/TRD, security, support, changelog, and release guidance.

For an already-built bundle, re-run `scripts/sales_readiness.py --require-acquisition-readiness` with every required artifact path explicitly. Missing buyer-facing evidence fails closed; it is not treated as skipped.

## Evidence Boundary

A successful acquisition-readiness manifest demonstrates that the configured software/procurement evidence profile is internally complete for the candidate. It does not establish:

- transaction value or valuation;
- customer acceptance, revenue, or legal transfer;
- deployment or operational availability;
- regulated/high-stakes suitability;
- psychometric validity, fairness, or downstream decision utility beyond the evidence actually supplied.

## Legacy 20B Compatibility

`scripts/build_commercial_release.py`, `--require-20b-product`, and `docs/20b_product_readiness.md` are retained for older automation that expects the historical 20B compatibility profile. They are not the generic acquisition gate. New buyer/release workflows must use `build_acquisition_release.py` / `--require-acquisition-readiness` and must supply any transaction scenario explicitly.

## Security and Support Boundaries

Security scope is documented in [`../SECURITY.md`](../SECURITY.md). Support scope is documented in [`../SUPPORT.md`](../SUPPORT.md). Both must match the exact package behavior being offered.

## Release Gate

Release candidates must not change the model formula, diagnostics semantics, or estimation scope without a separate model-design review. Packaging, docs, tests, and examples may change only while preserving the model contract and its evidence.

## Operational Notes

- Source and editable installs require a Rust toolchain because maturin builds `fast_mlsirm._core`.
- Installed wheels ship the compiled Rust core. `auto` uses that core and fails closed if it is missing. Pass `backend="numpy"` only for the explicit reference/parity path.
- The Rust backend is a dense-matrix backend, not a sparse storage layer.
- Real assessment data must remain under the buyer's privacy, governance, retention, and audit policies.
