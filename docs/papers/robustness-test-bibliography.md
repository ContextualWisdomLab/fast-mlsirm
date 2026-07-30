# IRT robustness-test bibliography (APA 7)

Full APA 7 references for the literature that grounds the IRT robustness
regression tests under `tests/`. Each test docstring cites these in-line; this
file is the consolidated, style-consistent bibliography. References are ordered
alphabetically by first author.

## References

Baker, F. B., & Kim, S.-H. (2004). *Item response theory: Parameter estimation
techniques* (2nd ed.). Marcel Dekker.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
443–459. <https://doi.org/10.1007/BF02293801>

Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor
analysis. *Applied Psychological Measurement, 12*(3), 261–280.
<https://doi.org/10.1177/014662168801200305>

Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods.
*Journal of Educational Measurement, 43*(4), 355–381.
<https://doi.org/10.1111/j.1745-3984.2006.00021.x>

Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
Methods and practices* (3rd ed.). Springer.
<https://doi.org/10.1007/978-1-4939-0317-7>

Lord, F. M. (1980). *Applications of item response theory to practical testing
problems*. Lawrence Erlbaum Associates.

Oakes, D. (1999). Direct calculation of the information matrix via the EM
algorithm. *Journal of the Royal Statistical Society: Series B (Statistical
Methodology), 61*(2), 479–482. <https://doi.org/10.1111/1467-9868.00188>

Penrose, R. (1955). A generalized inverse for matrices. *Mathematical
Proceedings of the Cambridge Philosophical Society, 51*(3), 406–413.
<https://doi.org/10.1017/S0305004100030401>

Pritikin, J. N. (2017). A comparison of parameter covariance estimation methods
for item response models in an expectation-maximization framework. *Cogent
Psychology, 4*(1), Article 1279435.
<https://doi.org/10.1080/23311908.2017.1279435>

Rothenberg, T. J. (1971). Identification in parametric models. *Econometrica,
39*(3), 577–591. <https://doi.org/10.2307/1913267>

van der Linden, W. J. (2005). *Linear models for optimal test design*.
Springer. <https://doi.org/10.1007/0-387-29054-0>

van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability
estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas (Eds.),
*Elements of adaptive testing* (pp. 3–30). Springer.
<https://doi.org/10.1007/978-0-387-85461-8_1>

## Reference → test map

| Reference | Robustness property | Test file |
|---|---|---|
| Rothenberg (1971); Penrose (1955) | Singular/rank-deficient information → Moore-Penrose pseudo-inverse vcov | `tests/test_information_matrix_robustness.py` |
| Lord (1980); Baker & Kim (2004) | Zero-/full-score (extreme raw score) fit-to-convergence stays finite | `tests/test_irt_extreme_score_stability.py` |
| Bock & Aitkin (1981); Kolen & Brennan (2014) | Concurrent multigroup MMLE robust to missing responses | `tests/test_concurrent_calibration_missing.py` |
| Bock, Gibbons & Muraki (1988) | Full-information item factor model stability (Heywood suppression) | `tests/test_full_information_factor_stability.py` |
| Oakes (1999); Pritikin (2017) | EM (Oakes-identity) standard errors; observed-information guards | `tests/test_information_matrix_robustness.py` |
| van der Linden & Pashley (2010); Lord (1980) | CAT maximum-Fisher-information selection tracks ability | `tests/test_cat_ata_information_robustness.py` |
| van der Linden (2005) | Automated test assembly is total-information optimal | `tests/test_cat_ata_information_robustness.py` |
| Kim (2006); Kolen & Brennan (2014) | Partial-anchor fixed-item-parameter linking places unique items | `tests/test_fixed_item_parameter_linking_robustness.py` |
