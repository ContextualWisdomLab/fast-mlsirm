# Recovery-validity references (APA 7)

Literature grounding for the estimator recovery-validity tests in
`tests/test_irt_stability.py`. These complement `mmle-lsirm-formula-compilation.md`
(which grounds the LSIRM/MLS2PLM *model* and estimator formulas): the references
below ground the *statistical validity* properties those tests assert about the
fitted estimator — parameter recovery, missing-response robustness, and
consistency. Each citation was verified against its publisher record; DOIs are
given where one exists.

## Test → grounding map

| Test in `tests/test_irt_stability.py` | Property asserted | Grounded in |
|---|---|---|
| `test_estimator_recovers_true_item_parameters_monte_carlo` | Monte-Carlo item-parameter recovery: estimates track a known truth better than a permutation null | Baker & Kim (2004) |
| `test_concurrent_calibration_is_robust_to_missing_responses` | Marginal/ML calibration stays consistent under data missing at random (observed-data likelihood ignores missing cells) | Rubin (1976); Mislevy & Wu (1996) |
| `test_item_parameter_recovery_improves_with_sample_size` | Estimator consistency: item-parameter recovery improves as N grows (sampling variability is O(1/√N)) | Lord (1980); Baker & Kim (2004) |

The pre-existing fixed-item-parameter calibration test
(`test_fipc_public_api_freezes_anchors_and_frees_population` in
`tests/test_scoring_methods.py`) is grounded in Kim (2006), included below for a
complete recovery/calibration bibliography.

## References

Baker, F. B., & Kim, S.-H. (2004). *Item response theory: Parameter estimation
techniques* (2nd ed.). Marcel Dekker.

Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods.
*Journal of Educational Measurement, 43*(4), 355–381.
<https://doi.org/10.1111/j.1745-3984.2006.00021.x>

Lord, F. M. (1980). *Applications of item response theory to practical testing
problems*. Lawrence Erlbaum Associates.

Mislevy, R. J., & Wu, P.-K. (1996). *Missing responses and IRT ability
estimation: Omits, choice, time limits, and adaptive testing* (ETS Research
Report No. RR-96-30). Educational Testing Service.
<https://doi.org/10.1002/j.2333-8504.1996.tb01708.x>

Rubin, D. B. (1976). Inference and missing data. *Biometrika, 63*(3), 581–592.
<https://doi.org/10.1093/biomet/63.3.581>
