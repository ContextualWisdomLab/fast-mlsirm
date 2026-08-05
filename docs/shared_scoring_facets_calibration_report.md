# Shared scoring-facets calibration report names

`fast_mlsirm.scoring.calibration_reporting` provides domain-neutral names for the
existing governed facets calibration report. The shared surface is an alias layer,
not a second report schema or estimator path:

```python
from fast_mlsirm.scoring.calibration_reporting import (
    ScoringFacetsCalibrationReport,
    build_scoring_facets_calibration_report,
    fit_scoring_facets_calibration_report,
)
```

The aliases are the exact same Python objects as the established
`fast_mlsirm.scoring.essay.calibration_reporting` entry points. This preserves
package ABI, report fingerprints, report handles, structured error identifiers,
and existing essay integrations while allowing enterprise issue and future domain
adapters to depend on a shared scoring namespace.

## Compatibility boundary

The current canonical implementation was initially named for automated essay
scoring. Consequently, legacy serialized handles and some structured error codes
retain `essay` in their stable wire identity. This slice does not rename those
values in place. A future rename requires an explicit schema version, migration,
rollback path, and dual-read compatibility period.

New domain adapters should use the shared import path. Existing essay callers may
continue using the essay names indefinitely. Both paths resolve to one class and
one pair of builder functions, so there is no duplicated validation, likelihood,
optimization, scoring, ranking, utility, or reporting arithmetic.

## Scientific boundary

The report remains a provenance and integrity artifact over an exact shared
`ScoringFacetsDesign` and copied Rust-backed `FacetsFit`. It does not establish
model adequacy, construct validity, fairness, scoreability, predictive validity,
rater interchangeability, global optimality, causal effects, or deployment
readiness. Domain-specific interpretation and human validation remain required.

## Verification

`tests/test_scoring_calibration_reporting.py` proves object identity between the
shared and essay names, explicit exports, inherited public documentation, and the
absence of a competing report class.
