# KRW 2,000,000,000 Benchmark Evidence Design

## Objective

Make benchmark evidence reviewable and gate-checkable for KRW 2,000,000,000
procurement review. The current product already records release-acceptance
command timing, but the evidence is buried inside `acceptance_summary.json`.
This wave turns that timing into `benchmark_report.json` and
`benchmark_report.html`.

## Boundaries

- Do not add a separate library, submodule, package split, hosted dashboard,
  frontend app, database, benchmark framework, or non-stdlib dependency.
- Do not use Figma Code Connect.
- Do not change model formulas, estimators, diagnostics, or interpretation
  semantics.
- Treat the report as release evidence, not a production performance guarantee.

## Product Design

The existing Figma file
`https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem` remains the visual
reference. The `02-synthetic-demo-run` frame maps to the generated benchmark
report, and `06-procurement-packet` maps to optional inclusion of
`benchmark/benchmark_report.json` and `benchmark/benchmark_report.html` in the
buyer packet.

The HTML report is static, keyboard-reviewable, and intentionally not a hosted
dashboard. It must include a CSP meta tag and focusable tables.

## Data Analytics

The report translates release acceptance timing into buyer-readable KPIs:

- runtime budget;
- total release-acceptance duration;
- budget pass/fail status;
- backend scenario coverage;
- per-command duration;
- required artifact coverage;
- benchmark caveats.

The authoritative timing source is `acceptance_summary.json`. The authoritative
scenario and artifact contract is `examples/enterprise_demo/benchmark_manifest.json`.

## Ponytail / Architecture Decision

No library split is appropriate. The feature is a narrow artifact generator and
readiness validator around existing release evidence. A new benchmark package
would add release, packaging, and review surface without improving the buyer
proof.

## Acceptance

- `scripts/build_benchmark_report.py` creates `benchmark_report.json` and
  `benchmark_report.html`.
- Sales readiness can validate benchmark report status, runtime budget,
  scenario coverage, artifact coverage, HTML existence, and HTML SHA256.
- Buyer packet generation can optionally include both benchmark report files.
- Documentation and manifests point to the generated benchmark report as actual
  evidence.
