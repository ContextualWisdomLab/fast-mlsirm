# 20B Commercial Release Builder Design

## Goal

Give a KRW 2,000,000,000 buyer-review run one command and one top-level
manifest. The command should coordinate the existing release evidence pipeline
without introducing a new package, submodule, hosted dashboard, or Figma Code
Connect workflow.

## Decision

Keep the work in the current repository as `scripts/build_commercial_release.py`.
This is the smallest useful product surface: it reuses existing scripts,
produces audit artifacts, and avoids adding versioning boundaries before there
is a separately consumed runtime or SDK.

The Figma buyer-review file remains evidence for the workflow. Its confirmed
frames are:

- `01-package-evidence`
- `02-synthetic-demo-run`
- `03-fit-diagnostics`
- `04-dimensionality-review`
- `05-report-export`
- `06-procurement-packet`

The builder maps those screens to stage output and does not use Figma Code
Connect.

## Evidence Contract

The builder must run or validate these stages:

- distribution build into the configured `--dist` directory, unless
  `--skip-build` is used;
- release acceptance;
- benchmark report;
- sales readiness;
- buyer evidence packet;
- release evidence index;
- final sales-readiness gate requiring benchmark, buyer packet, and release
  index evidence.

It must output:

- `commercial_release_manifest.json` for machine review;
- `commercial_release_report.html` for human procurement review.

The manifest must include source commit, generated timestamp, contract value,
stage commands, statuses, durations, stdout/stderr tails, failed stage, artifact
paths, artifact existence, sizes, and SHA256 digests.

The HTML report must include a restrictive CSP meta tag, a decision summary, a
focusable stage table, and a focusable artifact table.

## Analytics Scope

The builder records reproducibility and evidence coverage. It does not claim
valuation, regulated-use suitability, or buyer ROI. Those remain caveated in
the ROI evidence model and readiness docs.

## Go/No-Go

The commercial release run is `ok` only when every stage exits successfully and
the final sales-readiness manifest reports `status: ok`. If any stage fails,
the builder must stop, record `failed_stage`, still emit the manifest and HTML
summary, and exit non-zero from the CLI.

## Non-Goals

- New MLSIRM formulas, diagnostics semantics, or estimator scope.
- A hosted product surface, dashboard, login, billing, or customer-data upload.
- Artifact signing, package registry publication, or external attestation.
- Figma Code Connect.
