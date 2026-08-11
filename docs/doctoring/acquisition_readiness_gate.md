# Acquisition readiness gate (CLI)

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

## Rationale

Commercial readiness verification may validate an explicitly supplied deal scenario, but product quality evidence must not depend on a hard-coded monetary target. The public CLI therefore:

1. Defaults `--contract-value-krw` to unset.
2. Exposes `--require-acquisition-readiness` as a generic evidence-completeness gate.
3. Avoids help text that presents a single KRW 2B figure as a quality proof.

## Implementation

`scripts/sales_readiness.py` `build_parser()` and `run_sales_readiness()`.
