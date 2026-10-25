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

## Unfinished acceptance

This candidate still needs independent review of the repaired complete source,
full-library and canonical CI acceptance, final importer-base composition, and
primary-literature verification of interpretation-facing documentation. No claim
of full scientific adoption or selected-model adequacy is made. The retained
study-data execution is separate evidence and is not packaged here.
