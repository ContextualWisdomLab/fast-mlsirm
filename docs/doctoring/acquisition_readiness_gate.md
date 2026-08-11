# Acquisition readiness gate (CLI)

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association. https://www.testingstandards.net/open-access-files.html

The official open-access source covers professional and technical standards for
test development, validation, interpretation, and use. It is relevant here as
the measurement-quality basis for the acquisition evidence gate: a commercial
packet may demonstrate software and procurement readiness, but it cannot turn
those artifacts into validity, fairness, or high-stakes fitness claims without
the appropriate testing evidence.

## Rationale

Commercial readiness verification may validate an explicitly supplied deal scenario, but product quality evidence must not depend on a hard-coded monetary target. The public CLI therefore:

1. Defaults `--contract-value-krw` to unset.
2. Exposes `--require-acquisition-readiness` as a generic evidence-completeness gate.
3. Activates the buyer-packet, benchmark, release-evidence, procurement,
   PR-queue, and Figma evidence validators without requiring the legacy 20B
   product bundle.
4. Keeps `--require-20b-product` as a deprecated compatibility profile and
   records any explicitly supplied transaction scenario separately.

## Implementation

`scripts/sales_readiness.py` `build_parser()` and `run_sales_readiness()`.
