# Person-fit reporting metadata (schema 1)

This is an evidence-boundary correction, not a new numerical method or a
validation of the existing correction. Snijders (2001, pp. 333, 341) is now
directly checked in the primary PDF and page images. The paper explicitly
extends its results to bounded polytomous outcomes; a "dichotomous only"
exclusion is therefore not supported. The remaining HOLD concerns whether the
current EAP/category-log-likelihood implementation meets the theorem's
conditions and has recovery evidence. Access to this source is not the blocker.

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

## Primary-source check (2026-09-24)

Snijders, T. A. B. (2001). Asymptotic null distribution of person fit statistics
with estimated person parameter. *Psychometrika, 66*(3), 331–342.
https://doi.org/10.1007/BF02294437

- Source identity: Zotero personal library `users/0`, item `9FSBEPXY`, attachment
  `ZBFXFGNT`; PDF SHA256
  `6b1ccd2df19be89a2a778fa07d4ef92a93435de9c3429c5d93231999d60a4ddb`.
- Locators checked directly: PDF page 3 = printed 333, PDF page 11 = printed
  341 (offset +330 at both checked pages). Extracted text is checked against
  both page images, including equation (5).
- Page 333 requires an estimator satisfying equation (5); the examples include
  ML, Warm's estimator and "posterior mode estimators". It also states a
  nondegenerate root-n limit requirement. A posterior mean is not certified
  merely by adding the normal-prior score term to the correction.
- Page 341 states that the results "can be generalized to polytomous items".
  Its extension concerns linear statistics of bounded outcomes with the stated
  expectation, replacing the Bernoulli variance in equation (14) by the
  outcome variance. The partial-credit example uses ML satisfying equation (5).
  This establishes the extension's existence, not its automatic application
  to every GRM/GPCM category-log-likelihood statistic or EAP estimator.
- The same page limits the reported finite-sample simulation: smaller tail
  probabilities are liberal and skewness needs a more precise approximation.
  Those simulations do not calibrate this implementation's reporting cutoff.

The unresolved implementation evidence is the estimator-equation/theorem
mapping for the actual EAP and category-log-likelihood path, followed by
appropriate finite-sample/null-recovery verification. This bounded source
check does not prove impossibility, select another estimator, change a formula
or assign reporting acceptance. Schema 1's unverified states remain unchanged.
