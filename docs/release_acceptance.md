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

For a full buyer-review evidence bundle, prefer the single commercial release
builder:

```bash
python scripts/build_commercial_release.py \
  --out commercial-release \
  --require-rust \
  --check-import
```

The builder writes `commercial_release_manifest.json` and
`commercial_release_report.html`, then runs procurement due diligence by
default to write `procurement_due_diligence_manifest.json` and
`procurement_due_diligence_report.html`, and runs PR queue governance by
default to write `pr_queue_governance_manifest.json` and
`pr_queue_governance_report.html` under the same output directory. It then
runs Figma evidence sync by default to write
`figma_evidence_sync_manifest.json` and `figma_evidence_sync_report.html` after
PR queue governance so the static Figma buyer-review packet is checked against
the repo-local evidence bundle. It also
records each stage command, duration, status, failed stage, source commit,
contract value, artifact paths, and SHA256 digests for the same release
candidate.

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

Procurement due diligence can also be generated as a standalone stage when a
buyer asks for package metadata, policy-file, commercial-release, and GitHub
snapshot evidence:

```bash
python scripts/build_procurement_due_diligence.py \
  --dist dist \
  --commercial-release-manifest commercial-release/commercial_release_manifest.json \
  --out procurement-due-diligence
```

The procurement report writes `procurement_due_diligence_manifest.json` and
`procurement_due_diligence_report.html`. A final gate can require it with
`scripts/sales_readiness.py --procurement-due-diligence procurement-due-diligence/procurement_due_diligence_manifest.json --require-procurement-due-diligence`.

PR queue governance can also be generated as a standalone stage when a buyer
asks how open PRs, review delays, stale changes, and release-scope conflicts
are being managed:

```bash
python scripts/build_pr_queue_governance.py \
  --out pr-queue-governance
```

The PR queue report writes `pr_queue_governance_manifest.json` and
`pr_queue_governance_report.html`. A final gate can require it with
`scripts/sales_readiness.py --pr-queue-governance pr-queue-governance/pr_queue_governance_manifest.json --require-pr-queue-governance`.

Figma evidence sync can also be generated as a standalone stage when a buyer
asks whether the static design packet still reflects the same procurement
evidence being offered:

```bash
python scripts/build_figma_evidence_sync.py \
  --out figma-evidence-sync
```

The Figma sync report writes `figma_evidence_sync_manifest.json` and
`figma_evidence_sync_report.html`. A final gate can require it with
`scripts/sales_readiness.py --figma-evidence-sync figma-evidence-sync/figma_evidence_sync_manifest.json --require-figma-evidence-sync`.

## Required Rust Core

`--backend auto` requires the compiled Rust core and fails closed when that
extension is unavailable. Omitting `--require-rust` skips only the second,
explicit `--backend rust` fit; it does not enable a NumPy fallback for the
automatic production acceptance path. Explicit NumPy remains a reference and
parity choice outside this release-acceptance path.

## Linux Platform Floor and the NumPy Runtime Dependency

- fast-mlsirm Linux wheels target `manylinux2014` (glibc 2.17), as set in
  `.github/workflows/publish-pypi.yml`. The 0.11.5 cp312 x86_64 core was built in
  the pinned `manylinux2014` image and `auditwheel show` reports
  `manylinux_2_17_x86_64` (highest symbol `GLIBC_2.16`).
- NumPy is an external runtime dependency (`numpy>=1.24`). It is not bundled in
  fast-mlsirm wheels or the sdist, so its platform floor follows NumPy's own
  release policy: NumPy 2.5 publishes only `manylinux_2_27`/`manylinux_2_28`
  Linux wheels, and its `meson.build` requires GCC >= 10.3, which the
  `manylinux2014` toolchain (GCC 10.2.1) cannot provide. An environment with
  NumPy 2.5 therefore has an effective glibc floor of 2.27/2.28 regardless of
  the fast-mlsirm wheel.
- The official NumPy wheels vendor `libgfortran` and `libquadmath`. Because
  NumPy is installed by the user and not shipped by fast-mlsirm, these do not
  violate the rule against bundling GPL/LGPL/AGPL code in fast-mlsirm artifacts.
- Research consumption (A3) must use a Fortran-free NumPy build (OpenBLAS with
  `NOFORTRAN=1 C_LAPACK=1`, no `libgfortran`/`libquadmath` in the wheel) and
  record its wheel SHA256 next to the fast-mlsirm wheel SHA256 in the
  reproduction log.
