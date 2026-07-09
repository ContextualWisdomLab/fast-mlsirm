# IRT Stability Product Design and Equation Contract

## Purpose

`fast-mlsirm` is scientific software. Formula edits must preserve the declared
model contract unless a complete new model path is introduced. The current
MLS2PLM path remains the simple-structure specialization:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * distance(xi_p, zeta_i)
```

Do not replace this with the full discrimination-vector form through a local
optimization PR. A full MLS2PLM model requires a separate parameter shape,
likelihood, gradient, tests, docs, and Rust parity path.

## Literature Basis

- Bock and Aitkin (1981) ground marginal maximum-likelihood IRT estimation with
  EM and multiple latent dimensions:
  https://link.springer.com/article/10.1007/BF02293801
- Bock, Gibbons, and Muraki (1988) ground full-information item factor analysis
  with marginal maximum likelihood and omitted/not-reached response handling:
  https://journals.sagepub.com/doi/10.1177/014662168801200305
- `mirt::secondOrderTest` checks positive definiteness of a symmetric Hessian
  or information matrix:
  https://rdrr.io/cran/mirt/man/secondOrderTest.html
- `mirt` records SE/vcov/Hessian second-order checks in its IRT workflow:
  https://cran.hafro.is/web/packages/mirt/refman/mirt.html
- ATA tools commonly assemble forms from item metadata, constraints, and
  optimization/greedy selection surfaces:
  https://cran.r-project.org/package%3DmstATA
- Kang and Jeon (2025) document the multidimensional latent-space IRT context
  used by this repository's MLS2PLM scope:
  https://doi.org/10.1017/psy.2025.5

## Information Architecture

- Evidence source: simulated or buyer-provided response matrix, factor IDs,
  fitted parameters, optional mask, and optional content labels.
- Calibration stability: missing-by-design response acceptance, 0-score and
  full-score bounded initialization, finite objective/gradient checks.
- Equation stability: true-parameter probability reproduction, recovery report,
  observed information, vcov, standard errors, and second-order test.
- Test design support: fixed item parameter linking, CAT next-item selection,
  and ATA form assembly with content min/max constraints.
- Review output: static report/Figma frame that shows go/no-go status and the
  source command or API call for each stability check.

## Screen Definition

Screen `07-irt-stability-review` belongs after the existing diagnostics and
procurement screens.

- Primary task: verify that a release preserves scientific model stability.
- Inputs shown: response matrix status, mask status, factor IDs, parameter file,
  anchor item IDs, CAT/ATA constraints.
- Main panels: Missingness Robustness, True-Parameter Reproduction,
  Hessian/vcov/SE, Linking, CAT/ATA.
- Empty-state rule: no blank panels. If a check has no artifact, show a failed
  or not-run state with the expected API or command.
- Go/no-go signal: every panel has an explicit status, evidence path, and
  regression test name.

## Key Screen

The key screen is `07-irt-stability-review` because it connects the mathematical
contract to buyer-facing release confidence. It should be a dense evidence
screen, not a marketing hero.

## Wireframe

```text
+--------------------------------------------------------------+
| IRT Stability Review                              status: go |
| source commit | backend | response mask | model contract     |
+----------------------+----------------------+----------------+
| Missingness          | True Parameters      | Hessian / SE   |
| observed axes        | max |p_hat-p_true|   | min eigenvalue |
| all-missing rows     | recovery RMSE        | vcov finite    |
| 0/full-score guard   | regression test      | SE finite      |
+----------------------+----------------------+----------------+
| Fixed Item Linking   | CAT Selection        | ATA Assembly   |
| anchors used         | selected next item   | length         |
| scale / shift        | administered set     | content bounds |
| target metric check  | information score    | selected items |
+----------------------+----------------------+----------------+
| Evidence table: test name | API | artifact path | result       |
+--------------------------------------------------------------+
```

## User Stories

- As a psychometric researcher, I need missing-by-design rows and items to stay
  in the response matrix so concurrent calibration does not fail before model
  estimation.
- As a reviewer, I need true-parameter reproduction tests so formula changes
  cannot silently alter the simulation contract.
- As a statistician, I need Hessian/vcov/standard-error and second-order checks
  so a fitted solution has a visible local-stability signal.
- As a calibration engineer, I need anchor-item linking so fixed item
  parameters can put a new run onto an existing metric.
- As an assessment designer, I need CAT and ATA helpers so item information and
  content constraints can be exercised before a hosted testing product exists.

## Regression Contract

The release must preserve these tests:

- `tests/test_irt_stability.py::test_prepare_response_keeps_missing_by_design_axes`
- `tests/test_irt_stability.py::test_objective_and_diagnostics_are_finite_with_all_missing_axes`
- `tests/test_irt_stability.py::test_fit_handles_missing_by_design_axes_and_extreme_scores`
- `tests/test_irt_stability.py::test_true_parameters_reproduce_simulation_probabilities`
- `tests/test_irt_stability.py::test_hessian_vcov_standard_errors_and_second_order_check_are_stable`
- `tests/test_irt_stability.py::test_fixed_item_parameter_linking_recovers_anchor_metric`
- `tests/test_irt_stability.py::test_cat_item_selection_and_greedy_ata_constraints`
