# Add buyer-facing item-bank lifecycle reports

## Added

- Added deterministic JSON and standalone accessible HTML reporting for complete governed item-bank lifecycle lineages, including current state, rubric/blueprint provenance, approved-use scope, evidence-class inventory, transition timeline, and explicit missing-evidence limitations.
- Cross-version comparability is reported only as supported when governed linking evidence is present; the report never infers comparability from a nominal score range or active lifecycle state.
- Reporting remains provenance-only: calibration, fit, DIF, information, linking, exposure, drift, and uncertainty arithmetic are referenced by exact evidence identity and are not recomputed in Python.
