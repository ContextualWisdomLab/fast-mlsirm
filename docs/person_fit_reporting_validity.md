# Person-fit reporting metadata (schema 1)

This is an evidence-boundary correction, not a new numerical method or a
validation of the existing correction. The implementation's own method note
in `crates/mlsirm-core/src/poly.rs` records that the polytomous EAP correction
is an extrapolation whose original pinpoint is not verified. This patch does
not independently verify that literature. Method/reporting acceptance remains
on HOLD pending primary-source and recovery evidence.

Rust `PolyPersonFit`, the PyO3 `poly_person_fit` dictionary, the Python public
`compute_person_fit_polytomous` producer and its deprecated alias preserve
their numerical fields. The Python producer also preserves convergence,
termination and loaded-core provenance. Convergence refusal rules, priors,
quadrature arguments, thresholds and numerical formulas do not change.

Dictionary contract:

| Field | Current value | Meaning |
|---|---|---|
| validity_schema_version | integer 1 | Metadata protocol, not acceptance |
| valid_person_fit | false | Conservative compatibility flag for this corrected-result bundle |
| diagnostic_only | true | Corrected outputs cannot be promoted to validated reporting |
| statistic_validity.lz | not_assessed_uncorrected | Uncorrected descriptive statistic; reporting acceptance is not assessed here |
| statistic_validity.lz_star | unverified_polytomous_eap_correction | Correction is not validated for reporting |
| statistic_validity.flagged | unverified_polytomous_eap_correction | Threshold decision derives from the same unverified correction |

Rust exposes the corresponding `lz_validity` and `corrected_validity` fields;
PyO3 maps them into `statistic_validity`. The Python wrapper assigns these
conservative states even with an older core lacking metadata. Neither a caller
boolean nor a core's positive metadata can turn convergence into acceptance.
This does not assert that uncorrected `lz` shares the correction's defect.

Consumers must not infer reportability from finite values, successful return,
fit convergence or the presence of `flagged`. Absent, unknown, diagnostic-only
or unverified metadata must not publish corrected statistics or flags. Schema
version must be an integer, not truthy `True`. A future positive protocol value
`validated_for_research_reporting` is reserved for the selected statistic,
together with `valid_person_fit is True` and `diagnostic_only is False`.
**This producer issues no such positive value for any statistic.** Synthetic
consumer fixtures using that reserved value test only a future protocol path;
they do not validate a method or a release.

Verification scope: `tests/test_person_fit_validity_contract.py` loads the
actual Python producer and alias with a fake native routine; it performs no
numerical fit. Native compilation, PyO3 execution and the existing numerical
producer suite require separate verification and are not claimed by these
mock regressions.
