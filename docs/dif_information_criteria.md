# DIF information criteria

`fast_mlsirm.dif_information_criteria` is an additive relative-model-selection
boundary. It accepts two already fitted, converged models and never launches a
fit, changes anchors, purifies items, adjusts p-values, or reads study data.

## Inference contract

Use `dif_fit_evidence(fit, n_persons=..., comparison_key=...,
likelihood_kind="unpenalized_marginal_ml")` to retain the final
**observed marginal log-likelihood**, the native free-parameter count and the
convergence state. Untagged fits require the explicit `likelihood_kind`
attestation; recorded and supplied kinds must agree. Unknown fit objectives
are never silently called ML. The caller's nonempty comparison key must identify the same
response matrix, row order, category encoding, likelihood definition, estimator
and quadrature/QMC specification. Equal keys are an attestation by the caller,
not an independent data-identity proof. They must not be made from only N or a
requested device. Free-parameter counting includes estimated population terms
and excludes fixed/constrained duplicates; it is never inferred from stacked
parameter-array size or the likelihood-ratio df.

Call `compare_dif_information_criteria(compact, augmented, criterion="BIC")` with
an explicit criterion: `AIC`, `BIC`, `AICc`, `SABIC`, or `CAIC`. The existing
`mlsirm_core::fitstats::information_criteria` function supplies all formulas.
`delta_ic = IC(augmented) - IC(compact)`; a negative delta prefers the augmented
model with additional free parameters; exact ties keep compact. An undefined
AICc (`N <= K + 1`) cannot make a selection. Nonfinite likelihoods/criteria,
unconverged fits, mismatched provenance, and a non-increasing augmented
parameter count are errors. A negative likelihood improvement is retained; it
is not silently clamped or replaced with another fit.

This is a relative criterion preference, with no alpha, p-value, significance,
Holm, FWER or practical-importance guarantee. Holm adjustment has the different
goal of strong family-wise error control over a declared p-value family (R Core
Team, 2026, `p.adjust`, Details). IC differences are relative model comparisons,
not significance tests (Kang et al., 2009, pp. 500, 503).

`dif_fit_evidence` rejects slope-prior/MAP fits. The criterion must not consume a
penalized objective, log posterior or mean of the EM likelihood trace. An
observed likelihood evaluated at a MAP estimate is also not an ordinary
maximized-ML fit. Proper threshold priors and Bayesian inference need a
separately declared and validated contract; this API does not provide it.

## DIC and mirt scope

Pinned reference source: mirt commit
`3d28363f8524311e4724e5265c1e2966cc30315e`.

- `R/DIF.R` lines 69–73 documents AIC/SABIC/HQ/BIC and numeric p criteria;
  lines 270–275 sets `p.adjust='holm'`.
- The whitelist at line 438 contains `DIC`, but the actual anova producer
  (`R/SingleGroup-methods.R`, lines 587–605) has AIC/SABIC/HQ/BIC/logLik and no
  DIC column. Searching the pinned `R/` and `man/` trees finds `DIC` only in
  that whitelist and no `dIC` or `AICc` token. Whitelist admission is not an
  implemented criterion. No R fit is run as an oracle.
- DIF uses the compact-versus-augmented row order at line 351, tests the
  compact−augmented IC difference at lines 465–474, and returns the reverse
  difference at lines 610–616. `delta_ic` here explicitly names that latter
  **per-criterion difference**. It is not an alias named DIC.
- AICc/CAIC are fast-mlsirm's existing conventions, not claimed mirt DIF parity.
  HQ is not added because the reused Rust IC type does not supply it.

Bayesian **deviance information criterion** is
`DIC = mean_posterior(D) + p_D = D(posterior_mean) + 2*p_D`, where
`p_D = mean_posterior(D) - D(posterior_mean)` (Kang et al., 2009, p. 502,
eq. 4; Spiegelhalter et al., 2014, p. 486). Its likelihood focus and
parameterization matter (Spiegelhalter et al., 2014, pp. 487–488). The boundary
has no posterior-sample/deviance contract and rejects `DIC` explicitly. It never
substitutes an IC difference, twice a MAP parameter count, or an EM trace
average for posterior expectation.

## Execution scope

The scalar IC formulas, subtraction and selection are **Rust CPU binary64**.
They are valid consumers of GPU-computed likelihoods, but are not GPU kernels.
A device request alone does not prove that the likelihood E-step or item
M-step executed on a GPU. Native `gpu_execution_used`, backend/device and
fallback receipt fields, where exposed by a fit API, are retained separately.
Unknown provenance stays unknown. Existing receipt fields without stage detail
must not certify a GPU M-step. No CPU fallback is labelled GPU execution.

The new API is imported from the dedicated module, without rebinding historical
DIF APIs or taking ownership of sequential-controller, persistence or GPU
M-step modules. Integration of those lanes belongs to their integration owner.

## References

Kang, T., Cohen, A. S., & Sung, H.-J. (2009). Model selection indices for
polytomous items. *Applied Psychological Measurement, 33*(7), 499–518.
https://doi.org/10.1177/0146621608327800

Spiegelhalter, D. J., Best, N. G., Carlin, B. P., & van der Linde, A. (2014).
The deviance information criterion: 12 years on. *Journal of the Royal
Statistical Society: Series B, 76*(3), 485–493.
https://doi.org/10.1111/rssb.12062

R Core Team. (2026). *Adjust P-values for Multiple Comparisons* (`stats`,
`p.adjust`, Details).
https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html

Chalmers, R. P. (2026). *Differential item functioning statistics* (`mirt`,
`DIF`, source pinned above).
https://philchalmers.github.io/mirt/reference/DIF.html

Source definitions:
https://github.com/philchalmers/mirt/blob/3d28363f8524311e4724e5265c1e2966cc30315e/R/DIF.R
https://github.com/philchalmers/mirt/blob/3d28363f8524311e4724e5265c1e2966cc30315e/R/SingleGroup-methods.R
https://github.com/philchalmers/mirt/blob/3d28363f8524311e4724e5265c1e2966cc30315e/R/03-estimation.R
