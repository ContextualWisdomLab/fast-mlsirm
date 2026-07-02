# Release Acceptance Guide

## Purpose

`fast-mlsirm` is treated as commercially supportable only after this
release-acceptance smoke test passes on the exact release artifact or installed
package.

The script verifies:

- simulation output generation
- fitting with resolved backend (`--backend auto`)
- explicit Rust fit backend when requested
- fit and dimensionality diagnostics output
- diagnostics report HTML rendering

## How to Run

```bash
python scripts/release_acceptance.py \
  --out release_acceptance \
  --persons 12 \
  --dims 1 \
  --items-per-dim 2 \
  --latent-dim 1 \
  --max-iter 1 \
  --n-restarts 1 \
  --latent-dims 1,2 \
  --folds 2 \
  --require-rust
```

### Expected success shape

- Exit code `0`
- JSON result printed to stdout with `"status": "ok"`
- `acceptance_summary.json` written under `--out` with step outputs
- Generated artifacts:
  - `simulate/responses.npy`
  - `fit_auto/fit_summary.json`
  - `fit_rust/fit_summary.json` (present when `--require-rust`)
  - `diagnostics_fit/fit_diagnostics.json`
  - `diagnostics_dimensions/dimension_diagnostics.json`
  - `fit_report.html`
  - `dimension_report.html`

## Optional Local Mode

If Rust backend is unavailable in the local environment, run without
`--require-rust` and the script validates the default `auto` path only. For
commercial package verification, CI and distribution checks should keep
`--require-rust`.
