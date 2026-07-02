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
- per-step and total runtime evidence for sales-readiness review

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
- each step in `acceptance_summary.json` includes `duration_seconds`
- `acceptance_summary.json` includes `total_duration_seconds`
- Generated artifacts:
  - `simulate/responses.npy`
  - `fit_auto/fit_summary.json`
  - `fit_rust/fit_summary.json` (present when `--require-rust`)
  - `diagnostics_fit/fit_diagnostics.json`
  - `diagnostics_dimensions/dimension_diagnostics.json`
  - `fit_report.html`
  - `dimension_report.html`

## Enterprise Sales Gate

After building and installing the candidate package, run the enterprise sales
readiness verifier against the release-acceptance output:

```bash
python scripts/sales_readiness.py \
  --acceptance release_acceptance/acceptance_summary.json \
  --dist dist \
  --require-rust \
  --check-import \
  --out release_acceptance/sales_readiness_manifest.json
```

The command writes `sales_readiness_manifest.json`. A candidate is ready for
KRW 2,000,000,000 enterprise sales review only when every manifest check is
`ok`.

For the KRW 2,000,000,000 buyer packet flow, build the benchmark report,
buyer packet, and release evidence index from the same acceptance output:

```bash
python scripts/build_benchmark_report.py \
  --acceptance release_acceptance/acceptance_summary.json \
  --out release_acceptance/benchmark

python scripts/build_buyer_packet.py \
  --acceptance release_acceptance/acceptance_summary.json \
  --sales-readiness release_acceptance/sales_readiness_manifest.json \
  --dist dist \
  --benchmark-report release_acceptance/benchmark/benchmark_report.json \
  --out buyer-evidence-packet

python scripts/build_release_evidence_index.py \
  --acceptance release_acceptance/acceptance_summary.json \
  --sales-readiness release_acceptance/sales_readiness_manifest.json \
  --dist dist \
  --benchmark-report release_acceptance/benchmark/benchmark_report.json \
  --buyer-packet-manifest buyer-evidence-packet/buyer_evidence_manifest.json \
  --out release-evidence-index
```

The release index writes `release_evidence_index.json` and
`release_evidence_index.html`. It records the source commit, package version,
wheel and source distribution SHA256 digests, acceptance status, benchmark
budget status, sales-readiness status, buyer packet ZIP digest, and HTML report
digest. A final gate can require it with
`scripts/sales_readiness.py --release-evidence-index release-evidence-index/release_evidence_index.json --require-release-evidence-index`.

## Optional Local Mode

If Rust backend is unavailable in the local environment, run without
`--require-rust` and the script validates the default `auto` path only. For
commercial package verification, CI and distribution checks should keep
`--require-rust`.
