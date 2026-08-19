# Crossed multiple-membership person effects `u_h`

## Decision

`fast_mlsirm.multilevel.estimate_crossed_person_effects` estimates the
contextual random effects `u_h` of a binary IRT linear predictor when a
person belongs to more than one group at once. The kernel is Rust-owned.
Python validates, marshals a sealed `ContextMembershipDesign`, and reports
the immutable result. There is no Python numerical fallback and no stub.

The implemented predictor is

```text
eta_pi = a_i * (theta_p + sum_h w_ph * u_h) + b_i
```

`a_i` and `b_i` are known item parameters. `w_ph` are the Browne, Goldstein,
and Rasbash (2001) membership weights, already normalized to one inside each
classification. `theta_p` is an optional caller-supplied offset so a later or
already-estimated longitudinal state can enter the linear predictor. This
slice does not estimate OLS trends or AR coefficients.

## Scientific rationale

Fox and Glas (2001) specify a multilevel IRT model in which person location
depends on a cluster-level random effect with a Gaussian level-2 prior. The
ordinary nested case is one-hot membership: each person belongs to exactly
one unit of one classification. Browne, Goldstein, and Rasbash (2001) extend
the same additive random-effect term to multiple membership (several units of
one classification, weights summing to one) and multiple classification
(several classifications at once, i.e. crossed effects).

This kernel is the matching MAP / ridge point estimator of the flattened
effects `u_h`, not the Fox and Glas Gibbs sampler and not an MMMC MCMC
variance-component engine. The reported estimate is re-centered to sum to
zero inside each classification so recovered effects are deviations. A
classification with fewer than two levels is rejected because the location
constraint would leave a non-identified singleton.

## Compute boundary

The `O(n_persons * n_items)` Bernoulli score and information reduction is
multithreaded on CPU and, when a wgpu adapter is present, executed by an f32
GPU kernel. Missing adapters fall back to the f64 CPU path. Sparse membership
accumulation and the dense Newton system remain on CPU. `worker_count` does
not change the numerical result.

## Verification

Recovery evidence is RMSE against known simulated `u_h` under a crossed
school × neighborhood design that also includes weighted dual-school
membership. Correlation is supplementary only. Interval coverage,
variance-component ML, causal contextual effects, and continuous-time
dynamics are out of scope for this slice.

## APA 7th references

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
103–124. https://doi.org/10.1177/1471082X0100100202

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839
