# fast-mlsirm

`fast-mlsirm` is an early high-performance toolkit for Multidimensional
Latent Space Item Response Models, focused on MLS2PLM simulation, regularized
point estimation, and true-parameter recovery checks.

The first implementation keeps the public API small:

```python
import numpy as np

from fast_mlsirm import MLS2PLMConfig, FitConfig, fixed_item_calibration_diagnostics, simulate, dimensionality_diagnostics, fit, fit_diagnostics, recovery_report, render_diagnostics_report, response_process_dimensionality_diagnostics, response_process_fit_diagnostics

data = simulate(MLS2PLMConfig(seed=20260101))
result = fit(
    responses=data.Y,
    factor_id=data.factor_id,
    config=FitConfig(model="MLS2PLM", optimizer="adam_lbfgs", max_iter=100, backend="auto"),
)
report = recovery_report(data.truth, result.params)
diagnostics = fit_diagnostics(data.Y, result.params, data.factor_id, model=result.model)
dimensions = dimensionality_diagnostics(
    data.Y,
    data.factor_id,
    latent_dims=[1, 2, 3],
    config=FitConfig(model="MLS2PLM", optimizer="adam", max_iter=10, n_restarts=1),
)

print(report.summary)
print(diagnostics.model_fit)
print(dimensions.best)

category_probs = np.stack([1.0 - data.probabilities, data.probabilities], axis=2)
process_fit = response_process_fit_diagnostics(
    data.Y,
    category_probs,
    item_type="dichotomous",
    response_process="cumulative",
    group_id=np.arange(data.Y.shape[0]) % 2,
)
print(process_fit.itemfit["outfit_mnsq"])

process_dimensions = response_process_dimensionality_diagnostics(
    data.Y,
    {"dim2": category_probs},
    item_type="dichotomous",
    response_process="cumulative",
)
print(process_dimensions.best)

fixed_item_calibration = fixed_item_calibration_diagnostics(
    data.Y,
    {"dim2": category_probs},
    fixed_items=np.arange(min(4, data.Y.shape[1])),
    item_type="dichotomous",
    response_process="cumulative",
)
print(fixed_item_calibration.best)
```

## What Works Now

- Canonical MLS2PLM binary response simulation.
- `gamma=0` no-CD simulation.
- Regularized JML/MAP-style fitting for `MIRT`, `MLSRM`, `MLS2PLM`,
  `ULSRM`, and `ULS2PLM` constraints.
- Missing response exclusion via `NaN`, `-1`, or an explicit mask, including
  missing-by-design rows or items when at least one response is observed.
- Adam and small L-BFGS-style optimizers without SciPy.
- Procrustes alignment and distance-based recovery metrics.
- Point-estimate item, person, and model fit diagnostics for fitted models.
- K-fold held-out likelihood diagnostics for latent-space dimensionality.
- Shared dichotomous/polytomous response-process diagnostics from category
  probabilities.
- Multigroup and multilevel-context fit summaries from person-level group or
  cluster IDs.
- Response-process probability candidate comparisons for external dimensionality
  checks.
- True-parameter reproduction, observed-information Hessian, vcov, standard
  error, and second-order stability helpers.
- Fixed item parameter linking, CAT item-information selection, and greedy ATA
  form assembly with content min/max constraints.
- aFIPC-style fixed-item calibration diagnostics that select candidate
  probability tensors using fixed evaluation-item likelihood and kaefa-style
  item-fit penalty.
- Standalone HTML reports for saved fit or dimensionality diagnostics.
- Automated benchmark evidence reports from release-acceptance timing.
- Release evidence index reports that tie dist artifact hashes, acceptance,
  benchmark, sales-readiness, and buyer-packet evidence to one commit.
- Single-command commercial release evidence builder for dist, acceptance,
  benchmark, sales-readiness, buyer packet, release index, and final gate
  output.
- Procurement due-diligence evidence reports for distribution metadata,
  policy files, commercial-release integrity, GitHub snapshot state, and
  SHA256-verified HTML review output.
- PR queue governance evidence reports for open PR review state, stale and
  changes-requested risk counts, release-scope conflict classification, and
  SHA256-verified HTML review output.
- Figma evidence sync reports that verify the static buyer-review design packet
  still references buyer packet, release evidence index, procurement due
  diligence, and PR queue governance evidence while Code Connect stays disabled.
- CLI commands for simulation and fitting.
- Optional Rust-backed fitting objective via PyO3/maturin, with NumPy as the
  default reference backend.

## Install

For local development:

```bash
python -m pip install -e .
```

The default runtime backend is NumPy. Source and editable installs use maturin
to build the optional `fast_mlsirm._core` extension, so they require a working
Rust toolchain even if you later run with `backend="numpy"`. Installed wheels
can still use the NumPy default, and `backend="auto"` falls back to NumPy when
the extension is unavailable. The core Rust workspace can be tested with:

```bash
cargo test --workspace
```

The PyO3 extension crate is built by maturin and exercised by the Python backend
parity tests.

## Commercial Readiness

The current release is supportable as a commercial beta for technical teams that need
local MLS2PLM simulation, point-estimate fitting, diagnostics, and report
generation. It is not a regulated decision product, hosted assessment platform,
or Bayesian posterior inference engine. See:

- [Commercial readiness gate](docs/commercial_readiness.md)
- [Enterprise sales readiness gate](docs/enterprise_sales_readiness.md)
- [KRW 2,000,000,000 product readiness gate](docs/20b_product_readiness.md)
- [Buyer demo storyboard](docs/buyer_demo_storyboard.md)
- [Figma product design packet](docs/figma_product_design_packet.md)
- [IRT stability product design and equation contract](docs/irt_stability_product_design.md)
- [ROI evidence model](docs/roi_evidence_model.md)
- [Release acceptance guide](docs/release_acceptance.md)
- [Security policy](SECURITY.md)
- [Support policy](SUPPORT.md)
- [Changelog](CHANGELOG.md)

Sales readiness verification uses:

```bash
python scripts/build_commercial_release.py \
  --out commercial-release \
  --require-rust \
  --check-import
```

The commercial release builder writes `commercial_release_manifest.json` and
`commercial_release_report.html` while keeping the underlying stage artifacts
under the same output directory. The equivalent manual sequence is:

```bash
python scripts/release_acceptance.py --out acceptance_check --require-rust
python scripts/build_benchmark_report.py \
  --acceptance acceptance_check/acceptance_summary.json \
  --out acceptance_check/benchmark
python scripts/sales_readiness.py \
  --acceptance acceptance_check/acceptance_summary.json \
  --dist dist \
  --require-rust \
  --require-20b-product \
  --benchmark-report acceptance_check/benchmark/benchmark_report.json \
  --require-benchmark-report \
  --check-import \
  --out acceptance_check/sales_readiness_manifest.json
python scripts/build_buyer_packet.py \
  --acceptance acceptance_check/acceptance_summary.json \
  --sales-readiness acceptance_check/sales_readiness_manifest.json \
  --dist dist \
  --benchmark-report acceptance_check/benchmark/benchmark_report.json \
  --out buyer-evidence-packet
python scripts/build_release_evidence_index.py \
  --acceptance acceptance_check/acceptance_summary.json \
  --sales-readiness acceptance_check/sales_readiness_manifest.json \
  --dist dist \
  --benchmark-report acceptance_check/benchmark/benchmark_report.json \
  --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json \
  --out release-evidence-index
python scripts/sales_readiness.py \
  --acceptance acceptance_check/acceptance_summary.json \
  --dist dist \
  --require-rust \
  --require-20b-product \
  --benchmark-report acceptance_check/benchmark/benchmark_report.json \
  --require-benchmark-report \
  --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json \
  --require-buyer-packet \
  --release-evidence-index release-evidence-index/release_evidence_index.json \
  --require-release-evidence-index \
  --check-import \
  --out acceptance_check/final_sales_readiness_manifest.json
python scripts/build_procurement_due_diligence.py \
  --dist dist \
  --commercial-release-manifest commercial-release/commercial_release_manifest.json \
  --out procurement-due-diligence
python scripts/build_pr_queue_governance.py \
  --out pr-queue-governance
python scripts/build_figma_evidence_sync.py \
  --out figma-evidence-sync
python scripts/sales_readiness.py \
  --acceptance acceptance_check/acceptance_summary.json \
  --dist dist \
  --require-rust \
  --require-20b-product \
  --benchmark-report acceptance_check/benchmark/benchmark_report.json \
  --require-benchmark-report \
  --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json \
  --require-buyer-packet \
  --release-evidence-index release-evidence-index/release_evidence_index.json \
  --require-release-evidence-index \
  --procurement-due-diligence procurement-due-diligence/procurement_due_diligence_manifest.json \
  --require-procurement-due-diligence \
  --pr-queue-governance pr-queue-governance/pr_queue_governance_manifest.json \
  --require-pr-queue-governance \
  --figma-evidence-sync figma-evidence-sync/figma_evidence_sync_manifest.json \
  --require-figma-evidence-sync \
  --check-import \
  --out acceptance_check/final_procurement_sales_readiness_manifest.json
```

Enterprise Sales Readiness for KRW 2,000,000,000 procurement review requires
the release acceptance and sales-readiness commands to pass on the exact
release artifact. The 20B product gate adds
Product Design, Figma-without-Code-Connect, Data Analytics, ROI, benchmark, and
synthetic demo evidence from `examples/enterprise_demo/`. The buyer packet
command produces a portable zip, `buyer_evidence_manifest.json`, and
`buyer_evidence_report.html` for procurement review. The benchmark command
produces `benchmark_report.json` and `benchmark_report.html` from the same
release-acceptance timing evidence. The release evidence index command produces
`release_evidence_index.json` and `release_evidence_index.html` as a compact
digest map over the candidate wheel, source distribution, release acceptance,
benchmark report, sales-readiness manifest, and buyer packet.
The commercial release builder produces the same evidence as a single buyer
review entrypoint and records the failed stage when the gate does not pass.
It now also invokes `scripts/build_procurement_due_diligence.py` by default and
emits `procurement_due_diligence_manifest.json` plus
`procurement_due_diligence_report.html` under the commercial release output.
It also invokes `scripts/build_pr_queue_governance.py` by default and emits
`pr_queue_governance_manifest.json` plus `pr_queue_governance_report.html` so
open GitHub PRs are inventoried as managed queue evidence rather than treated
as an unexamined release risk. It then invokes
`scripts/build_figma_evidence_sync.py` by default and emits
`figma_evidence_sync_manifest.json` plus `figma_evidence_sync_report.html` so
the static Figma procurement frame is checked against the same repo-local
buyer evidence packet without using Figma Code Connect.

## CLI

```bash
fast-mlsirm simulate \
  --persons 500 \
  --dims 2 \
  --items-per-dim 8 \
  --latent-dim 2 \
  --phi 0.3 \
  --gamma 1.5 \
  --seed 20260101 \
  --out runs/sim_001

fast-mlsirm fit \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --model MLS2PLM \
  --backend auto \
  --latent-dim 2 \
  --optimizer adam_lbfgs \
  --max-iter 100 \
  --out runs/fit_001

fast-mlsirm diagnose-fit \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --params runs/fit_001/params.npz \
  --model MLS2PLM \
  --out runs/diagnostics_001

fast-mlsirm diagnose-dimensions \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --latent-dims 1,2,3 \
  --folds 5 \
  --model MLS2PLM \
  --max-iter 100 \
  --out runs/dimensions_001

fast-mlsirm diagnose-response-process \
  --responses runs/sim_001/responses.npy \
  --probabilities runs/model_probabilities.npy \
  --item-type polytomous \
  --response-process cumulative \
  --group-id runs/group_id.npy \
  --cluster-id runs/school_id.npy \
  --out runs/process_fit_001

fast-mlsirm diagnose-response-candidates \
  --responses runs/sim_001/responses.npy \
  --candidate dim1=runs/prob_dim1.npy \
  --candidate dim2=runs/prob_dim2.npy \
  --item-type dichotomous \
  --response-process ideal_point \
  --out runs/process_dimensions_001

fast-mlsirm diagnose-fixed-item-calibration \
  --responses runs/sim_001/responses.npy \
  --candidate dim1=runs/prob_dim1.npy \
  --candidate dim2=runs/prob_dim2.npy \
  --fixed-items runs/fixed_items.npy \
  --item-type dichotomous \
  --response-process ideal_point \
  --itemfit-penalty-weight 1.0 \
  --out runs/fixed_item_calibration_001

fast-mlsirm render-report \
  --diagnostics runs/diagnostics_001/fit_diagnostics.json \
  --out runs/diagnostics_001/report.html
```

For automation, every CLI command also accepts `--json`. In JSON mode,
progress text is suppressed and stdout contains one status object with the
output directory, key metrics, and generated file paths:

```bash
fast-mlsirm simulate \
  --persons 500 \
  --dims 2 \
  --items-per-dim 8 \
  --out runs/sim_001 \
  --json

fast-mlsirm fit \
  --responses runs/sim_001/responses.npy \
  --factors runs/sim_001/item_factor.csv \
  --out runs/fit_001 \
  --json
```

`fit`, `diagnose-fit`, and `diagnose-dimensions` validate that `responses.npy`
is a 2D persons-by-items matrix and that `item_factor.csv` has exactly one
factor id per item before running optimization or diagnostics.
`diagnose-fixed-item-calibration` writes `dimension_diagnostics.json` with
`best_candidate`, `calibration_score`, fixed-item coverage counts, and
kaefa-style item-fit penalty metrics. `--fixed-items` accepts a `.npy` boolean
mask or item-index vector; when omitted, all items are treated as the fixed
calibration set.

`fit --backend numpy` uses the Python reference objective. `fit --backend rust`
requires the installed `fast_mlsirm._core` extension and fails clearly if it is
unavailable. `fit --backend auto` uses the Rust objective when available and
falls back to NumPy otherwise.

`render-report` turns `fit_diagnostics.json` or `dimension_diagnostics.json`
into a standalone HTML report with model summary cards, compact tables, and
small bar views when chartable diagnostic metrics are present. Optional fit
tables, dimensionality candidate comparisons, or metric summaries without
values are summarized in a diagnostics coverage block instead of rendering as
repeated blank-looking report sections or placeholder-only columns.

## Repository Layout

```text
python/fast_mlsirm/       Python public API and reference backend
crates/mlsirm-core/       Rust likelihood and gradient core
crates/fast-mlsirm-py/    PyO3 binding for the optional Rust backend
tests/                    Python smoke and numerical tests
docs/                     PRD/TRD summary and roadmap
examples/enterprise_demo/ Synthetic procurement evidence manifests
```

## MVP Boundary

This is not a Bayesian sampler. The package intentionally starts with fast
simulation, regularized point estimation, and recovery diagnostics. The current
Rust backend keeps the same point-estimate formula contract as the NumPy
reference path. Block-mode Rust execution, sparse response storage, benchmark
automation, posterior predictive checking, and new ordinal response estimators
remain future work.
