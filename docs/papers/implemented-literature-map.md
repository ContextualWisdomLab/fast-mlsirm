# Implemented-literature map

Where each paper of the supplied reading set landed in the codebase. The
implementation-ready extractions live in this repository at
`docs/papers/group_a_specs.md`, `docs/papers/group_b_specs.md`, and
`docs/papers/group_c_specs.md`; the estimator/scoring foundations are in
`docs/papers/mmle-lsirm-formula-compilation.md`.

| Paper | Status | Where |
|---|---|---|
| Schneider, Chalmers, Debelak & Merkle (2019), Vuong tests for IRT model selection, MBR | implemented (non-nested z + Schwarz correction; distinguishability pre-test documented-only) | `fitstats.rs::vuong_nonnested`, `fast_mlsirm.vuong_nonnested` |
| Pritikin (2017), EM parameter covariance comparison, Cogent Psychology | implemented (the recommended Oakes-identity estimator) | `oakes.rs`, `fast_mlsirm.oakes_standard_errors` |
| Kang, Cohen & Sung (2009), model-selection indices, APM | implemented (AIC/BIC/AICc/SABIC/CAIC + free-parameter counting; BIC the default comparator; DIC/CVLL documented-only — Bayesian) | `fitstats.rs::information_criteria`, `FitResult.ic` |
| Dueber (2021), `BifactorIndicesCalculator`; Rodriguez, Reise & Haviland (2016), methodological origin | implemented (continuous-indicator ECV-SS/SG/GS, item ECV, strict-pattern PUC, omega total, omega hierarchical, H, plus explicitly named logistic latent-response standardization; no categorical-score omega claim and no hard-coded scoreability cutoff) | `bifactor_indices.rs`, `docs/bifactor_scoreability_indices.md`, `tests/bifactor_indices.rs`, `tests/bifactor_logit_indices.rs` |
| Svetina & Levy (2014), dimensionality-assessment framework, Educational Assessment | implemented (residual procedures: Yen Q3 + GDDM); DETECT/DIMTEST/NOHARM out of scope | `fitstats.rs::dimensionality_residuals`, `fast_mlsirm.dimensionality_residuals` |
| Sinharay & Lu (2008), item parameters vs item-fit correlation, JEM | implemented (S-X² plus S-G² grouped-score item-fit outputs; chi-square-G remains intentionally unimplemented) | `fitstats.rs::s_x2`, `fast_mlsirm.fitstats.s_x2` |
| Perumean-Chaney et al. (2013), zero-inflated/overdispersed count models, JSCS | implemented (structural-zero mixture for the marginal estimator; boundary-aware pi) | `marginal.rs` (`MarginalConfig.zero_inflation`), `FitConfig(zero_inflation=True)` |
| Jeon, Rijmen & Rabe-Hesketh (2013), multiple-group bifactor DIF, — | adapted (group-specific virtual items + anchors + LR); the current full-information bifactor fit and scoreability diagnostics do not yet implement their complete multiple-group bifactor DIF model | `fast_mlsirm.dif_analysis`, `ModelType::Bifac2plm`, `bifactor_indices.rs` |
| Debeer & Janssen (2013), item-position effects in IRT | implemented (linear position effect as a context-varying item covariate with estimated delta; person-specific random slope documented as upgrade path) | `marginal.rs::ItemCovariate`, `fit(covariate=...)` |
| Jeon & De Boeck (2016), generalized IRTree, BRM | implemented (mapping-matrix pseudo-item expansion; their Eq. 9 reduces IRTrees to binary IRT on the expanded matrix) | `fast_mlsirm.irtree_expand` |
| Huo et al. (2015), hierarchical multi-unidimensional IRT for sparse multi-group data | largely already-covered (simple-structure multidim + multigroup + MAR missingness + anchoring); free cross-dim covariance and hierarchical shrinkage of group means documented as future adaptations | `marginal.rs` multigroup path |
| Williamson, Xi & Breyer (2012), automated-scoring evaluation framework, EM:IP | implemented (QWK/r/SMD/degradation/subgroup conjunctive gates with the paper's thresholds) | `agreement.rs`, `fast_mlsirm.validate_judge` |
| Makransky & Glas (2013), group-specific item parameters for CAT DIF, Measurement | adapted (LR/Wald-style screen via virtual items; the LM statistic documented-only) | `fast_mlsirm.dif_analysis` |
| Ferrando, Lorenzo-Seva & Chico (2009), factor-analytic response-bias procedure, SEM | adapted (full tridimensional MRFA remains out of scope; added observed-minus-expected leniency residual proxy with explicit non-identification boundaries) | `diagnostics.py::fit_diagnostics` (`personfit.leniency_*`, `model_fit.leniency_*`), group C spec |
| Wolkowitz & Skorupski (2013), MC option imputation, EPM | superseded (marginal-ML integrates over missing cells under MAR; option-level imputation needs polytomous data) | group C spec |
| Joubert et al. (2015), forced-choice vs Likert psychometrics, IJSA | documented-only (Thurstonian forced-choice blocks absent from binary judge data) | group C spec |

## Additional implemented-method bibliography

These rows record primary sources for kernels that already exist on protected
main but were outside the original supplied reading-set table. Changelog
"NOT READ" notes and unmerged PRs are not the bibliographic record.

| Paper | Status | Where |
|---|---|---|
| Jeon, Jin, Schweinberger & Baugh (2021), LSIRM interaction map, Psychometrika; DOI [10.1007/s11336-021-09762-5](https://doi.org/10.1007/s11336-021-09762-5) | implemented (simple-structure MLSIRM/MLS2PLM specialization; not the full discrimination-vector model) | `docs/papers/mls2plm-canonical-equations.md`, ADR-0001, `fit` / `simulate` MLS2PLM path |
| Kang & Jeon (2025), multidimensional LSIRM / conditional-dependence relativity, Psychometrika; DOI [10.1017/psy.2025.5](https://doi.org/10.1017/psy.2025.5) | implemented (simple-structure MLS2PLM; residual interaction after substantive structure) | ADR-0001, ADR-0006, `docs/traceability/research-basis.md` |
| Molenaar & Jeon (2026), regularized JML for LSIRM, Psychometrika; DOI [10.1017/psy.2025.10068](https://doi.org/10.1017/psy.2025.10068) | implemented (penalized joint ML / MAP-style point estimation) | `docs/papers/mls2plm-canonical-equations.md`, ADR-0001 |
| Angoff (1972) APA paper; Angoff & Ford (1973) JEM; Magis & Facon (2012, 2014) BJMSP / JSS | implemented (Angoff TID / delta-plot with Magis–Facon threshold; not MH/logistic/SIBTEST; not a CWE/OWASP/NIST control) | `fast_mlsirm.delta_plot`, `mlsirm_core::dif::delta_plot`, `docs/delta_plot_dif.md`, ADR-0016 |
| Bradley & Terry (1952), paired comparisons, Biometrika; Hunter (2004), MM algorithms, Ann. Statist. | implemented (tie-free BT MM; additive-`alpha0` BRATT ties variant only — not Rao–Kupper/Davidson) | `fast_mlsirm.bradley_terry_mm`, `fast_mlsirm.bratt_mm`, `docs/bradley_terry_mm.md`, ADR-0017 |
| AERA, APA, & NCME (2014), *Standards for educational and psychological testing* | governing score-interpretation / fairness standard for the methods above | `docs/standards_watch.md`, ADR-0001, ADR-0016, ADR-0017 |
