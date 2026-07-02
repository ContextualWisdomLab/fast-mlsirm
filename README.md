# fast-mlsirm

`fast-mlsirm` is an early high-performance toolkit for Multidimensional
Latent Space Item Response Models, focused on MLS2PLM simulation, regularized
point estimation, and true-parameter recovery checks.

The first implementation keeps the public API small:

```python
import numpy as np

from fast_mlsirm import MLS2PLMConfig, FitConfig, simulate, dimensionality_diagnostics, fit, fit_diagnostics, recovery_report, render_diagnostics_report, response_process_dimensionality_diagnostics, response_process_fit_diagnostics

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
```

## What Works Now

- Canonical MLS2PLM binary response simulation.
- `gamma=0` no-CD simulation.
- Regularized JML/MAP-style fitting for `MIRT`, `MLSRM`, `MLS2PLM`,
  `ULSRM`, and `ULS2PLM` constraints.
- Missing response exclusion via `NaN`, `-1`, or an explicit mask.
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
- Standalone HTML reports for saved fit or dimensionality diagnostics.
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

## Commercial Beta Status

The current release is supportable as a commercial beta for technical teams that
need local MLS2PLM simulation, point-estimate fitting, diagnostics, and report
generation. It is not a regulated decision product, hosted assessment platform,
or Bayesian posterior inference engine. See:

- [Commercial beta readiness](docs/commercial_readiness.md)
- [Security policy](SECURITY.md)
- [Support policy](SUPPORT.md)
- [Changelog](CHANGELOG.md)

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
```

## MVP Boundary

This is not a Bayesian sampler. The package intentionally starts with fast
simulation, regularized point estimation, and recovery diagnostics. The current
Rust backend keeps the same point-estimate formula contract as the NumPy
reference path. Block-mode Rust execution, sparse response storage, benchmark
automation, posterior predictive checking, and new ordinal response estimators
remain future work.
