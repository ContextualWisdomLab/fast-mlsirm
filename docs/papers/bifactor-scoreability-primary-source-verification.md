# Bifactor scoreability primary-source verification

## Verification status

The complete author-posted text of Rodriguez, Reise, and Haviland (2016a), the publisher's correction, and the bibliographic record and abstract for Rodriguez, Reise, and Haviland (2016b) were reviewed on 2026-08-03. The correction is load-bearing: it replaces the printed forms of Equations 1, 4, and 7. Implementations and review evidence must use the corrected equations, not the uncorrected article display.

This record verifies the continuous standardized-loading contracts used by the Rust diagnostic. It does not convert descriptive indices into model-selection rules and does not establish universal cutoffs.

## Model scope

The article assumes a confirmatory bifactor loading matrix with:

- one general factor loading on every item;
- orthogonal general and group factors;
- standardized items;
- group factors representing content clusters; and
- no group-factor cross-loadings in the reviewed examples.

`bifactor_indices` therefore fails closed when the declared general factor is incomplete. It permits cross-loaded specific-factor matrices only as an explicit implementation extension: PUC is unavailable for those matrices, and the result must not be represented as the strict-bifactor PUC contract from the article.

## Equation traceability

### Omega total and omega hierarchical

Rodriguez et al. (2016a), article PDF page 2, Equations 1 and 3, define total-score omega and omega hierarchical from squared sums of standardized loadings and residual variances. The correction to Equation 1 replaces the duplicated general-factor term with the sum across group-factor loading columns.

For a strict bifactor item set, the Rust denominator is:

`sum_f (sum_i lambda_if)^2 + sum_i uniqueness_i`.

The total numerator includes every common-factor column sum; the hierarchical numerator retains only the target factor's squared loading sum. This is the corrected Equation 1 / Equation 3 structure.

### Omega hierarchical subscale

Rodriguez et al. (2016a), article PDF page 2, Equation 4, defines the reliable variance attributable uniquely to one group factor in a subscale. The publisher correction replaces a duplicated general-factor term in the denominator with the relevant group-factor term.

For a strict bifactor group-factor domain, the Rust target-factor numerator and domain denominator reproduce the corrected Equation 4. For an explicitly cross-loaded extension, all active common-factor column sums are retained in the denominator; that extension is not claimed to be Equation 4 verbatim.

### Construct replicability H

Rodriguez et al. (2016a), article PDF page 7, Equation 6, defines construct replicability as:

`H = 1 / (1 + 1 / sum_i(lambda_i^2 / (1 - lambda_i^2)))`.

The Rust `construct_replicability` calculation uses this expression independently for each factor column.

### Explained common variance

Rodriguez et al. (2016a), article PDF page 8, Equation 7, defines general-factor ECV as the general factor's sum of squared standardized loadings divided by the sum of squared loadings across the general and every group factor. The correction adds the explicit sum over all group factors.

The general-factor element of Rust `ecv_sg` reproduces the corrected global ECV expression. `ecv_ss`, group-factor `ecv_sg`, `ecv_gs`, and item ECV are extended decomposition indices traced to the complete `BifactorIndicesCalculator` 0.2.2 implementation; they are not misrepresented as equations printed in Rodriguez et al. (2016a).

### Percentage of uncontaminated correlations

Rodriguez et al. (2016a), article PDF pages 8-9, defines PUC conceptually as the proportion of item correlations influenced only by the general factor. This requires a strict bifactor pattern in which each item belongs to no more than one group factor. The Rust API returns `None` when specific-factor cross-loadings make that counting contract ambiguous.

## Interpretation boundary

The primary article discusses empirical interpretation ranges, but this repository does not hard-code them as pass/fail thresholds. Scoreability diagnostics are post-fit descriptive evidence. Selection among bifactor, correlated-traits, higher-order, testlet, two-tier, latent-space, or other models requires likelihood, predictive, recovery, invariance, and appropriate non-nested-comparison evidence.

The logistic-slope entry point is a separate latent-response transformation using orthogonal unit-variance factors and logistic residual variance `pi^2 / 3`. Its omega values describe the standardized continuous latent-response representation, not categorical observed-score reliability.

## References

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016a). Applying bifactor statistical indices in the evaluation of psychological measures. *Journal of Personality Assessment, 98*(3), 223-237. https://doi.org/10.1080/00223891.2015.1089249

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016b). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*(2), 137-150. https://doi.org/10.1037/met0000045

Taylor & Francis. (2016). Correction to: Applying bifactor statistical indices in the evaluation of psychological measures. *Journal of Personality Assessment, 98*(4), 444. https://doi.org/10.1080/00223891.2015.1117928

Dueber, D. M. (2021). *BifactorIndicesCalculator: Bifactor indices calculator* (Version 0.2.2) [R package]. https://CRAN.R-project.org/package=BifactorIndicesCalculator
