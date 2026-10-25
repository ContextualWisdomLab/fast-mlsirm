# Same-fit grouped GRM coefficient transformation (candidate)

This Rust API transforms an existing general-only, multiple-group logit GRM.
It does not optimize or select a model. `samefit_grm::PackedArchive` is supplied
by the separate generic importer change; importing equality does not establish
convergence, optimality or scientific acceptance.

For `logit P(Y >= k) = a * theta + d[k]`, the implemented representation is:
- difficulty: `-d[k] / a`;
- group latent-response loading: `a * sigma / sqrt((a*sigma)^2 + pi^2/3)`;
- standardized threshold: `-(d[k] + a*mu) / sqrt((a*sigma)^2 + pi^2/3)`.
Reference-group mean and SD are fixed at zero and one. Shared item coordinates
are reused across groups; free items retain group-specific coordinates. These
are algebraic representations of the same logit model, not a probit fit.

`transform` returns CPU f64 estimates and an analytic Jacobian. `infer` accepts
observed negative-log-likelihood information, requires positive definiteness,
and uses the existing native inverse to propagate the full covariance. Its
normal-Wald intervals are conditional on the supplied anchor mask, with no
post-selection adjustment. The information matrix is not an M2 covariance.

With feature `gpu`, `gpu::evaluate` takes parameter covariance (NOT information),
performs three f32 compute dispatches (transform/Jacobian, J*C, J*C*J^T), and
returns f64 containers of f32 results. CPU or unknown adapter types are refused;
backend labels alone are insufficient. Validation includes finite inputs,
positive-definite symmetric covariance, normal-range exponentials, preserved
f32 intercept ordering, checked shape arithmetic and storage/dispatch limits.
The positive-definiteness check also applies to the actual f32-rounded covariance;
a covariance that loses rank or becomes indefinite in transfer is refused.
CPU inference rejects a nonfinite inverse even when there are no display rows.
These checks do not guarantee accuracy for every condition number or scale.
There is no numerical CPU fallback. Covariance validation is CPU work.
The CPU information inverse and SE/CI are not GPU computations.

Tests use declared implementation tolerances, not literature-derived scientific
acceptance thresholds. Mixed-anchor tests compare every Jacobian coordinate to
finite differences and every covariance cell to an independent double contraction.

## Method basis and limits

Muthén and Asparouhov (2002, pp. 2–5, Eqs. 1–9 and 16–19) give the
conditional-probability and latent-response parameterizations and the response
mean/variance decomposition. Here the explicit implementation convention is
`L = a*theta + epsilon`, with independent standard-logistic `epsilon`, and
`Y >= k` exactly when `L >= -d[k]`. The logistic CDF has scale one and variance
`pi^2/3` (R Core Team, n.d., “Details”). Substituting `theta = mu + sigma*z`
gives response mean `a*mu`, variance `(a*sigma)^2 + pi^2/3`, standardized
slope `a*sigma / response_sd`, and boundary `(-d[k]-a*mu) / response_sd`.
The displayed formulas are this algebraic convention. They do not assert that
the marginal response is normal or that normal-ogive and logit fits are identical.
No conversion factor such as 1.7 is used. The reference metric and anchor mask
are caller/model choices, not evidence of measurement invariance.

For the analytic Jacobian `J` at the supplied packed estimates and full parameter
covariance `C`, the first-order propagation is `J C J^T`. This is the matrix
propagation formula of Joint Committee for Guides in Metrology (2011, p. 15,
§6.2.1.3, Eq. 3), with its sensitivity matrix equal to `J` and input covariance
equal to `C`; its first-order limitation is explicit in §5.6.1. This citation
supports covariance arithmetic, not likelihood optimality or anchor selection.
The normal-Wald construction follows the large-sample linearization described by
Xu and Long (2005, pp. 554–556, §7.2, Eq. 2). Their printed example is scalar;
the matrix form here is supported separately by JCGM. Each interval uses the
standard-normal 97.5th percentile and is pointwise, conditional on the fixed
model/anchor mask. It is neither simultaneous nor selection-adjusted and may
cross the natural bounds of a transformed coefficient. A finite positive
information matrix alone does not establish the required statistical regularity.

GPU covariance admission uses a FIFA-local Cholesky check on both the original
f64 covariance and the actual f32 values converted back exactly to f64. Each
matrix is divided by its largest absolute entry before factorization. All pivots
must be finite and strictly positive (Higham, 2009, p. 251, Eq. 1; p. 253).
The positive rescaling preserves definiteness in exact arithmetic and avoids the
shared eigensolver's absolute off-diagonal cutoff for small-scale inputs. No
shared inference function is changed. Near the definiteness boundary this remains
a floating-point decision, with the backward-error limitation discussed by
Higham (2009, p. 252, Eq. 4); it is not a condition-number or accuracy certificate.
The scale normalization, strict pivot policy, symmetry tolerance, shape caps,
normal-range exponential guard and regression tolerances are implementation
choices. These sources do not prescribe those numeric limits.

## References

Higham, N. J. (2009). Cholesky factorization. *Wiley Interdisciplinary Reviews:
Computational Statistics, 1*(2), 251–254. https://doi.org/10.1002/wics.18

Joint Committee for Guides in Metrology. (2011). *Evaluation of measurement
data—Supplement 2 to the “Guide to the expression of uncertainty in measurement”—
Extension to any number of output quantities* (JCGM 102:2011).
https://doi.org/10.59161/JCGM102-2011

Muthén, B., & Asparouhov, T. (2002). *Latent variable analysis with categorical
outcomes: Multiple-group and growth modeling in Mplus* (Mplus Web Notes No. 4,
Version 5). https://statmodel.com/download/webnotes/CatMGLong.pdf

R Core Team. (n.d.). *The logistic distribution* (R stats documentation, version
4.6.0, “Details”). https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Logistic.html

Xu, J., & Long, J. S. (2005). Confidence intervals for predicted outcomes in
regression models for categorical outcomes. *The Stata Journal, 5*(4), 537–559.
https://doi.org/10.1177/1536867X0500500405

## Unfinished acceptance

This candidate still needs independent review of the repaired complete source,
full-library and canonical CI acceptance, final importer-base composition, and
primary-literature verification of interpretation-facing documentation. No claim
of full scientific adoption or selected-model adequacy is made. The retained
study-data execution is separate evidence and is not packaged here.
