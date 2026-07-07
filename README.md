# fast-mlsirm

`fast-mlsirm` is an early high-performance toolkit for Multidimensional
Latent Space Item Response Models, focused on MLS2PLM simulation, regularized
point estimation, and true-parameter recovery checks.

The first implementation keeps the public API small:

```python
from fast_mlsirm import MLS2PLMConfig, FitConfig, simulate, fit, recovery_report

data = simulate(MLS2PLMConfig(seed=20260101))
result = fit(
    responses=data.Y,
    factor_id=data.factor_id,
    config=FitConfig(model="MLS2PLM", optimizer="adam_lbfgs", max_iter=100),
)
report = recovery_report(data.truth, result.params)

print(report.summary)
```

## What Works Now

- Canonical MLS2PLM binary response simulation.
- `gamma=0` no-CD simulation.
- Regularized JML/MAP-style fitting for `MIRT`, `MLSRM`, `MLS2PLM`,
  `ULSRM`, and `ULS2PLM` constraints.
- Missing response exclusion via `NaN`, `-1`, or an explicit mask.
- Adam and small L-BFGS-style optimizers without SciPy.
- Procrustes alignment and distance-based recovery metrics.
- CLI commands for simulation and fitting.
- Rust core crate with the same likelihood and gradient formulas.

## Install

For local development:

```bash
python -m pip install -e .
```

The Python reference backend requires NumPy. The Rust crate can be tested with:

```bash
cargo test
```

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
  --latent-dim 2 \
  --optimizer adam_lbfgs \
  --max-iter 100 \
  --out runs/fit_001
```

## Repository Layout

```text
python/fast_mlsirm/       Python public API and reference backend
crates/mlsirm-core/       Rust likelihood and gradient core
tests/                    Python smoke and numerical tests
docs/                     PRD/TRD summary and roadmap
```

## MVP Boundary

This is not a Bayesian sampler. The package intentionally starts with fast
simulation, regularized point estimation, and recovery diagnostics. PyO3
zero-copy bindings, block-mode Rust execution, sparse response storage, and
benchmark automation are next-step work after the formulas and API stabilize.
