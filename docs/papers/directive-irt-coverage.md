# Directive IRT requirements — literature grounding and coverage map

This document maps the psychometric/IRT capability requirements (full-information
item factor model stability, true-parameter recovery, SE/Hessian/vcov/second-order
preservation, concurrent-calibration missing-value robustness, fixed-item-parameter
linking, CAT, ATA, and zero/perfect-score robustness) to (a) their primary-source
literature basis in APA 7th-edition form, (b) the exact formula or invariant a test
should assert, and (c) where each is already implemented and regression-tested in
this repository.

It is the in-repository preservation of the source grounding (the ecosystem's
Zotero local API is not reachable from CI/headless environments, so durable
grounding lives here alongside `implemented-literature-map.md`,
`mls2plm-canonical-equations.md`, and `mmle-lsirm-formula-compilation.md`). No
Git LFS is used; primary sources are preserved as citations with DOIs and, where
open access exists, a stable OA URL for the full text.

## Coverage summary

Every target below is already implemented and covered by an executable regression
in `tests/`; this document records the literature each assertion rests on. It does
not add new behavior — it grounds existing behavior.

| Directive target | Primary source(s) | Implemented symbol | Regression test |
|---|---|---|---|
| Full-information item-factor stability | Bock & Aitkin (1981); Bock, Gibbons & Muraki (1988) | marginal estimator (`fit`, MMLE path) | `tests/test_estimator_mmle.py`, `tests/test_marginal_parity.py` |
| True-parameter recovery | Harwell, Stone, Hsu & Kirisci (1996); Reckase (2009) | `recovery_report` | `test_irt_stability.py::test_true_parameters_reproduce_simulation_probabilities` |
| SE = TRUE — Hessian / vcov / second-order | Oakes (1999); Cai (2008); Chalmers (2012) | `observed_information`, `vcov_from_hessian`, `standard_errors_from_vcov`, `oakes_standard_errors`, `second_order_test` | `test_irt_stability.py::test_hessian_vcov_standard_errors_and_second_order_check_are_stable` |
| Concurrent-calibration missing-value robustness | Rubin (1976); Bock & Aitkin (1981) | NaN / `-1` / mask handling in `fit`/objective | `test_irt_stability.py::test_*_missing_by_design_axes`, `test_estimator_marginal.py` (MAR NaNs) |
| Fixed-item-parameter linking (FIPC) | Kim (2006); Stocking & Lord (1983); Haebara (1980) | `link_fixed_item_parameters`, `irt_link`, `fixed_item_calibration_diagnostics` | `test_diagnostics.py`, FIPC diagnostics |
| CAT — selection / ability / stopping | van der Linden & Pashley (2010); Bock & Mislevy (1982) | Fisher-information selection + EAP/MLE scoring | scoring tests (`tests/unit/scoring_wle_tests.rs`), CAT paths |
| ATA — optimal test assembly | van der Linden (2005) | target-information assembly | assembly/paper-feature tests |
| Zero / perfect-score robustness | Warm (1989); Bock & Mislevy (1982) | Warm WLE + EAP boundary handling | `test_irt_stability.py::test_fit_handles_missing_by_design_axes_and_extreme_scores`, `tests/unit/scoring_wle_tests.rs` |

## Dedicated end-to-end regressions added alongside this document

The tables above map directive targets to pre-existing coverage. This branch also
adds two focused pytest modules that exercise the property end to end through the
public API (`simulate` → `fit` → `recovery_report`) and pin the missing-value
numeric contract on both backends. They add coverage, not behavior — no model
formula, objective, gradient, or Rust/NumPy numeric path is changed.

| Directive target | Primary source(s) | Dedicated regression |
|---|---|---|
| True-parameter recovery (marginal ML) | Bock & Aitkin (1981); Harwell et al. (1996); Reckase (2009) | `tests/test_parameter_recovery.py::test_marginal_estimator_recovers_generating_2pl_item_parameters` |
| Full-information item-factor stability | Bock, Gibbons & Muraki (1988); Bock & Aitkin (1981) | `tests/test_parameter_recovery.py::test_marginal_item_factor_solution_is_stable_across_run_configurations` |
| Zero / perfect / constant-item robustness | Warm (1989); Baker & Kim (2004) | `tests/test_missing_and_extreme_robustness.py::test_zero_and_perfect_score_persons_yield_finite_fit`, `::test_constant_items_yield_finite_objective` |
| Missing NaN/`-1`/mask equivalence + backend parity | Rubin (1976); Bock & Aitkin (1981) | `tests/test_missing_and_extreme_robustness.py::test_missing_sentinels_are_equivalent_and_masked_entries_do_not_contribute`, `::test_rust_and_numpy_agree_on_masked_inputs` |

These are complementary to the parallel test PRs on sibling branches (recovery and
0/full-score robustness): the modules here use the marginal (full-information)
estimator path and additionally pin the three-way missing-sentinel equivalence and
Rust↔NumPy masked-input parity, which the sibling PRs do not assert.

## Formula / invariant each test asserts

1. **Full-information item-factor stability (Bock & Aitkin, 1981; Bock, Gibbons &
   Muraki, 1988).** Item parameters maximize the *marginal* likelihood, integrating
   the latent trait out over its population distribution by Gauss–Hermite
   quadrature via EM (E-step forms expected node-wise response frequencies; M-step
   is reweighted item-level ML). Assertion: estimation operates on the marginal
   (not joint) likelihood, and the latent metric is fixed (mean 0, variance 1 /
   fixed rotation) so the solution is identified.

2. **True-parameter recovery (Harwell et al., 1996; Reckase, 2009).** No universal
   fixed threshold exists; the standard is the *design*: generate from known
   parameters, re-estimate, and report bias (mean of `θ̂ − θ`) and
   RMSE (`sqrt(mean((θ̂ − θ)²))`), asserting both shrink toward 0 with N
   (consistency) after resolving MIRT rotational/reflection indeterminacy
   (Procrustes/orthogonal alignment). `recovery_report` on identical truth returns
   `distance_rmse < 1e-12` and zero γ error.

3. **SE = TRUE — Hessian / vcov / second-order (Oakes, 1999; Cai, 2008; Chalmers,
   2012).** In an EM fit the observed-data information is recovered in closed form
   from derivatives of the EM Q-function (Oakes identity): complete-data curvature
   minus the missing-information term; SEs are `sqrt(diag(I⁻¹))`. The
   **second-order test** asserts the information/Hessian is positive definite at the
   solution (all eigenvalues > 0) — a genuine local maximum, not a saddle;
   `second_order_test` computes exactly this on the symmetrized Hessian.

4. **Concurrent-calibration missing-value robustness (Rubin, 1976; Bock & Aitkin,
   1981).** Under MAR with the missingness parameter distinct from the item
   parameters, the mechanism is *ignorable*: each examinee's likelihood is a product
   over answered items only, so MML over observed responses is consistent and
   unbiased and no imputation is required. Assertion: a fit on MAR-inserted NaNs
   recovers complete-data estimates within Monte-Carlo error; omitted cells
   contribute nothing to the likelihood; the objective and diagnostics stay finite
   under missing-by-design axes.

5. **Fixed-item-parameter linking (Kim, 2006; Stocking & Lord, 1983; Haebara,
   1980).** FIPC keeps anchor items' base-scale parameters fixed during MML
   re-estimation of the new form, updating the ability prior across EM cycles to
   absorb group difference; the new items land directly on the base metric.
   Assertion: anchor-item parameters are unchanged post-fit. Characteristic-curve
   alternatives (Stocking–Lord test-level, Haebara item-level) find `(A, B)` under
   `b* = Ab + B`, `a* = a/A`.

6. **CAT (van der Linden & Pashley, 2010; Bock & Mislevy, 1982).** Selection
   administers `argmax_j I_j(θ̂)`, with 2PL Fisher information
   `I_j(θ) = a_j² P_j(θ) Q_j(θ)`; ability by MLE (`Σ a_j[u_j − P_j(θ̂)] = 0`) or the
   always-finite EAP; fixed-precision stopping when
   `SE(θ̂) = 1/sqrt(Σ_j I_j(θ̂)) ≤ τ`.

7. **ATA (van der Linden, 2005).** Assembly is a 0–1 linear program selecting items
   `x_i ∈ {0,1}` to drive the test information function `I(θ_k) = Σ_i x_i I_i(θ_k)`
   toward a target `T(θ_k)` subject to content, length (`Σ_i x_i = n`), exposure,
   and enemy-set constraints. Assertion: the assembled form satisfies all
   constraints and its TIF meets the target within solver tolerance.

8. **Zero / perfect-score robustness (Warm, 1989; Bock & Mislevy, 1982).** For an
   all-0 or all-correct pattern the MLE score equation has no finite root (MLE
   diverges to ∓∞). Remedies: WLE solves the weighted score equation
   `∂lnL/∂θ + J(θ)/(2 I(θ)) = 0` (removes the O(1/n) bias, finite for extreme
   patterns); Bayesian EAP/MAP with a proper prior is always finite. Assertion:
   all-0/all-1 vectors return finite `θ̂` and defined SE; MLE is flagged/undefined.

## Open-access full-text availability (for future PDF preservation)

Preserve these OA copies under `docs/papers/papers/` if/when full-text archiving is
performed (no Git LFS; direct PDF or a citation-only stub per repository policy):

- Chalmers (2012) — fully OA: <https://www.jstatsoft.org/v48/i06/>
- Bock & Aitkin (1981) — OA scan (UC Merced course reserves)
- Bock, Gibbons & Muraki (1988) — OA (U. Minnesota Conservancy)
- Bock & Mislevy (1982) — OA (U. Minnesota Conservancy)
- Stocking & Lord (1983) — OA (U. Minnesota Conservancy)
- Rubin (1976) — OA author copy (Harvard DASH)
- Oakes (1999), Cai (2008), Kim (2006), Warm (1989), Harwell et al. (1996),
  Reckase (2009), van der Linden (2005) — publisher paywalled; cite by DOI.

## APA-7 references

Baker, F. B., & Kim, S.-H. (2004). *Item response theory: Parameter estimation techniques* (2nd ed.). Marcel Dekker.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459. https://doi.org/10.1007/BF02293801

Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor analysis. *Applied Psychological Measurement, 12*(3), 261–280. https://doi.org/10.1177/014662168801200305

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement, 6*(4), 431–444. https://doi.org/10.1177/014662168200600405

Cai, L. (2008). SEM of another flavour: Two new applications of the supplemented EM algorithm. *British Journal of Mathematical and Statistical Psychology, 61*(2), 309–329. https://doi.org/10.1348/000711007X249603

Cai, L. (2010). High-dimensional exploratory item factor analysis by a Metropolis–Hastings Robbins–Monro algorithm. *Psychometrika, 75*(1), 33–57. https://doi.org/10.1007/s11336-009-9136-x

Cai, L. (2010). Metropolis–Hastings Robbins–Monro algorithm for confirmatory item factor analysis. *Journal of Educational and Behavioral Statistics, 35*(3), 307–335. https://doi.org/10.3102/1076998609353115

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software, 48*(6), 1–29. https://doi.org/10.18637/jss.v048.i06

Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research, 22*(3), 144–149. https://doi.org/10.4992/psycholres1954.22.144

Harwell, M., Stone, C. A., Hsu, T.-C., & Kirisci, L. (1996). Monte Carlo studies in item response theory. *Applied Psychological Measurement, 20*(2), 101–125. https://doi.org/10.1177/014662169602000201

Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement, 43*(4), 355–381. https://doi.org/10.1111/j.1745-3984.2006.00021.x

Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188

Reckase, M. D. (2009). *Multidimensional item response theory*. Springer. https://doi.org/10.1007/978-0-387-89976-3

Rubin, D. B. (1976). Inference and missing data. *Biometrika, 63*(3), 581–592. https://doi.org/10.1093/biomet/63.3.581

Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement, 7*(2), 201–210. https://doi.org/10.1177/014662168300700208

van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer. https://doi.org/10.1007/0-387-29054-0

van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas (Eds.), *Elements of adaptive testing* (pp. 3–30). Springer. https://doi.org/10.1007/978-0-387-85461-8_1

Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika, 54*(3), 427–450. https://doi.org/10.1007/BF02294627
