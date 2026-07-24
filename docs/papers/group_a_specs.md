# Group A — Implementation-Ready Specs for fast-mlsirm

Target engine: **fast-mlsirm** (Rust core `mlsirm-core` + Python wrapper `fast_mlsirm`).
Estimation: marginal maximum likelihood (MMLE) by EM with Bock–Aitkin quadrature.
Person latents: trait `theta_p in R^D` (simple structure) + latent-space `xi_p in R^K`.
Item params: `(alpha, b, zeta, tau)`. Populations: single / multigroup / multilevel / FIPC.
Items: **binary only** for now.

What the engine already has (verified in source), which shapes every feasibility call below:

- `python/fast_mlsirm/diagnostics.py` already computes **AIC** (`2p − 2·loglik`) and **BIC** (`log(N)·p − 2·loglik`), plus MLE cross-validated held-out log-likelihood.
- `python/fast_mlsirm/inference.py` already exposes `observed_information` (finite-difference Hessian of the penalized negative log-likelihood — this is Pritikin's "central-difference full-parameter Hessian" benchmark), `vcov_from_hessian`, `standard_errors_from_vcov`, `second_order_test`.
- `python/fast_mlsirm/fitstats.py` + `crates/mlsirm-core/src/fitstats.rs` already implement **S-X²** (Orlando & Thissen 2000) with the Lord–Wingersky recursion on the `(theta, xi)` grid, plus `l_z`/`l_z*` and infit/outfit.
- `crates/mlsirm-core/src/marginal.rs` implements the EM map `M(theta)`, per-person posterior over quadrature nodes, and per-item M-step gradients — the exact primitives Oakes/SEM/Vuong need.

Bottom line up front:

| Paper | Core contribution | Feasibility for fast-mlsirm |
|---|---|---|
| 2 Pritikin | **Oakes' identity** for the observed-information matrix in EM | **Direct** — highest value; upgrades existing FD-Hessian SEs |
| 1 Schneider et al. | Vuong tests (distinguishability / non-nested / nested) | **Adaptation** — needs casewise scores + eigenvalue machinery |
| 3 Kang–Cohen–Sung | AIC/BIC/DIC/CVLL for model selection; BIC recommended | **Direct (AIC/BIC already done)**; DIC/CVLL document-only |
| 4 Svetina–Levy | Taxonomy/framework for dimensionality assessment | **Mostly document-only**; Q3 / GDDM / parallel analysis adaptable |
| 5 Sinharay–Lu | S-X² has clean Type-I error, no spurious item-param correlation | **Document-only** — validates the existing S-X² choice |

---

## Paper 2 — Pritikin (2017): Parameter covariance estimation in an EM framework

**Full citation.** Joshua N. Pritikin (2017). *A comparison of parameter covariance estimation methods for item response models in an expectation-maximization framework.* **Cogent Psychology** 4: 1279435. DOI: 10.1080/23311908.2017.1279435. (Open access, CC-BY.)

**What it is.** A Monte-Carlo bake-off of methods that recover the parameter covariance matrix `V` (hence standard errors `SE = sqrt(diag(V))`) from an EM fit, which does *not* produce `V` natively. Contestants: the completed-data (M-step) Hessian, central-difference + Richardson extrapolation, the Supplemented-EM family (MR-SEM, Tian-SEM, Agile-SEM), and **Oakes' direct method**. Across four IFA models (`m2pl5`, `m3pl15`, `grm20`, `cyh1`) Oakes wins on accuracy (KL divergence, `||RD||_2`) *and* elapsed time, and never fails to converge, whereas MR-SEM/Tian-SEM fail on a large fraction of `grm20`/`cyh1` trials.

### Exact estimating equations

Let `L(theta | Y_o)` be the observed-data likelihood, `Y_m` the "made-up" latent data (examinee latent scores), `Y_c = (Y_o, Y_m)` the completed data.

Complete-data (M-step) Hessian — asymptotically *under*estimates variability:

$$
\mathcal{H}_c(\hat\theta; Y_c) \;=\; -\,\frac{\partial^2 \log L(\theta \mid Y_c)}{\partial\theta\,\partial\theta^\top}\bigg|_{\hat\theta}
\tag{1}
$$

Observed-data information — the target, usually hard to evaluate directly:

$$
\mathcal{H}_o(\hat\theta; Y_o) \;=\; -\,\frac{\partial^2 \log L(\theta \mid Y_o)}{\partial\theta\,\partial\theta^\top}\bigg|_{\hat\theta}
\tag{2}
$$

**Finite differences + Richardson extrapolation** (Jamshidian & Jennrich 2000). For a scalar function `f`, central second difference:

$$
f''(\theta) \;\approx\; \frac{f(\theta-\delta) - 2f(\theta) + f(\theta+\delta)}{\delta^2}, \qquad \delta > 0
\tag{3}
$$

Richardson shrinks `delta` each iteration and extrapolates the curvature change. Cost `= 1 + r(N^2 + N)` likelihood evaluations (`r` iterations, `N` parameters) — quadratic in `N`, so only practical for small models. (fast-mlsirm's `inference.observed_information` is exactly this family, forward/central FD over the *full* parameter Hessian.)

**Missing-information principle** (Orchard & Woodbury 1972; Louis 1982). Completed information = observed + missing:

$$
\mathcal{I}(\theta; Y_c) \;=\; \mathcal{I}(\theta; Y_o) + \mathcal{I}(\theta; Y_m)
\tag{4a}
$$
$$
\big[\, I - \mathcal{I}(\theta; Y_m)\,\mathcal{I}(\theta; Y_c)^{-1} \,\big]\,\mathcal{I}(\theta; Y_c) \;=\; \mathcal{I}(\theta; Y_o)
\tag{5}
$$

**Supplemented EM** (Meng & Rubin 1991). One EM cycle is a map `theta_{t+1} = M(theta_t)`. Its Jacobian at the MLE,

$$
\mathrm{D}M \;=\; \frac{\partial M(\theta)}{\partial\theta}\bigg|_{\theta=\hat\theta},
\qquad
\mathrm{D}M \;\approx\; \mathcal{I}(\theta; Y_m)\,\mathcal{I}(\theta; Y_c)^{-1}
\tag{7,8}
$$

is the fraction of information the missing data contributes. Combining (5) and (8):

$$
V^{-1} \;=\; \mathcal{I}(\theta; Y_o) \;\approx\; \big(I - \mathrm{D}M\big)\,\mathcal{I}(\theta; Y_c).
$$

`DM` column `j` by forward-differencing the EM map (run one EM cycle with all params frozen at `theta_hat` except the `j`-th perturbed):

$$
r_{ij}(\theta_j) \;=\; \frac{M_i(\hat\theta_1,\dots,\hat\theta_{j-1},\,\theta_j,\,\hat\theta_{j+1},\dots,\hat\theta_d) - M_i(\hat\theta)}{\theta_j - \hat\theta_j}
\tag{9}
$$

Column `j` declared converged when `|r_ij(theta_t) − r_ij(theta_{t+1})| < tol` for all `i`, with `tol = sqrt(EM tolerance)` (10). MR-SEM/Tian-SEM/Agile-SEM differ only in *which* trajectory points `theta_t` seed (9) — Tian-SEM uses the near-convergence subset where `t_hat ∈ [.9, .999]`. These are the failure-prone parts.

**Oakes' direct method** (Oakes 1999) — the recommended method. It gives `I(theta; Y_m)` (equivalently the observed information) directly, without a convergence trajectory. Paper's form: the missing information is the Jacobian of the completed-data gradient w.r.t. the made-up data,

$$
\mathcal{I}(\theta; Y_m) \;=\; \frac{\partial}{\partial Y_m}\!\left[\frac{\partial \log L(\theta \mid Y_o, Y_m)}{\partial\theta}\right].
\tag{11}
$$

Implementation-canonical (equivalent) form — the one to code, since it is written in terms of the EM objective `Q` the engine already evaluates. With `Q(theta | theta_tilde) = E_{z | Y_o, theta_tilde}[ log L(theta; Y_o, z) ]` (the E-step expected complete-data log-likelihood),

$$
\boxed{\;
-\,\frac{\partial^2 \log L(\theta;Y_o)}{\partial\theta\,\partial\theta^\top}\bigg|_{\hat\theta}
=\;
-\left(
\underbrace{\frac{\partial^2 Q}{\partial\theta\,\partial\theta^\top}}_{\text{M-step Hessian (1)}}
+\;
\underbrace{\frac{\partial^2 Q}{\partial\theta\,\partial\tilde\theta^\top}}_{\text{cross term}}
\right)\bigg|_{\theta=\tilde\theta=\hat\theta}
\;}
$$

The cross term is obtained by forward-differencing the **M-step gradient** `g(theta, theta_tilde) = ∂Q/∂theta` with respect to the *conditioning* parameter `theta_tilde` — i.e. re-run the E-step at each perturbed `theta_tilde`, requiring only `N + 1` gradient evaluations (paper used forward difference, step `1e-5`). This is precisely statement (11).

### Quality measures used by the paper

Relative difference of SEs, and its `l2` summary:
$$
\mathrm{RD} = \frac{\mathrm{SE} - \mathrm{SE}_{\text{true}}}{\mathrm{SE}_{\text{true}}}, \qquad \|\mathrm{RD}\|_2.
$$
KL divergence between the Monte-Carlo "true" covariance and the estimate (zero-mean MVN, dimension `K`):
$$
D_{KL}(\Sigma_{\text{true}}, \hat\Sigma) = \tfrac{1}{2}\!\left[\operatorname{Tr}(\hat\Sigma^{-1}\Sigma_{\text{true}}) - K - \log\frac{|\Sigma_{\text{true}}|}{|\hat\Sigma|}\right].
$$

### Recommendation (explicit in the paper)

**Use Oakes.** It matched or beat every competitor on both accuracy and speed (except the low-accuracy raw M-step benchmark), never failed to converge, and is `N+1` evaluations (linear) vs Richardson's `1 + r(N²+N)`. On `cyh1` Oakes took 0.83 s vs Richardson's 46.4 s at equal accuracy; MR-SEM failed 70% of `cyh1` and 95% of `grm20` trials. The paper's closing argument: because Oakes is implemented optimally from theory, "the deciding factor... may be the parsimony of the theory," and Oakes is the most parsimonious. Caveat it raises: for parameters near a boundary, prefer profile-likelihood CIs over any Wald/`sqrt(diag V)` SE.

### Implementation plan (mapped to the marginal-EM engine)

E-step quantities needed (all already produced per EM cycle in `marginal.rs`):
- per-person posterior weights over `(theta, xi)` quadrature nodes at the current params;
- the M-step gradient of the expected complete-data log-likelihood `∂Q/∂(alpha,b,zeta,tau, population moments)` — already computed for the GEM gradient-ascent M-step.

New computation:
1. At the converged `theta_hat`, assemble the **complete-data Hessian** `∂²Q/∂theta²` (M-step Hessian). For items this is block-diagonal per item; for population moments it is closed-form. Cheap.
2. Compute the **Oakes cross term**: for each free parameter `j`, perturb `theta_tilde_j = theta_hat_j + eps` (eps ≈ `1e-5`), re-run **one E-step** at that `theta_tilde`, recompute the M-step gradient `g(theta_hat, theta_tilde)`, and forward-difference: column `j` of the cross term `= [g(theta_hat, theta_tilde) − g(theta_hat, theta_hat)] / eps`. `N+1` E-step + gradient passes.
3. Observed information `= −(M-step Hessian + cross term)`; symmetrize `(A + Aᵀ)/2`; `V = information⁻¹`; `SE = sqrt(diag V)`.

Wire it in as a new `method="oakes"` branch alongside the existing FD path in `inference.observed_information` / `vcov_from_hessian`, reusing the Rust E-step. The engine already exposes the FD full-Hessian, so this is an additive, drop-in-comparable estimator.

Computational cost: `O(N)` E-step passes (one per free parameter) + one Hessian assembly, vs the current full FD Hessian at `O(N²)` likelihood evaluations. For a 73-parameter GRM the paper saw ~100× speedup at equal or better accuracy.

Minimal correct test: on a small fixed 2PL data set (e.g. `m2pl5`-style, 5 items, `N=1000`), assert Oakes SEs match the existing central-difference `observed_information` SEs to within a few percent `||RD||_2`, and assert the Oakes information matrix is symmetric positive-definite. A stronger optional test: on a simulated data set with a known Monte-Carlo covariance, assert `log D_KL(Oakes) ≤ log D_KL(FD)`.

### Not implementable / out of scope

- SEM family (MR/Tian/Agile-SEM): implementable in principle (the EM map `M` exists), but the paper's own evidence is that they are slower and fail to converge far more often — **skip**; Oakes dominates. `ponytail:` don't build the losers.
- Nominal/graded-model covariance results generalize per the paper, but fast-mlsirm is **binary only**, so only the dichotomous parameterization applies today.
- Profile-likelihood CIs (the paper's boundary-case recommendation) are a separate, heavier feature — out of scope for this spec.

---

## Paper 1 — Schneider, Chalmers, Debelak & Merkle (2019): Vuong tests for IRT model selection

**Full citation.** Lennart Schneider, R. Philip Chalmers, Rudolf Debelak & Edgar C. Merkle (2019). *Model Selection of Nested and Non-Nested Item Response Models Using Vuong Tests.* **Multivariate Behavioral Research.** DOI: 10.1080/00273171.2019.1664280.

**What it is.** Applies Vuong's (1989) three tests — (i) **distinguishability**, (ii) **non-nested goodness-of-fit**, (iii) **nested** — to marginal-ML IRT models, so both nested and non-nested models get a *formal statistical test* (not just an information criterion). Implemented as an extension of R `nonnest2` driving `mirt` fits. The tests make **no assumption that either model is correctly specified** — their edge over the classical LR test in misspecified / different-dimension comparisons.

### Exact statistics

Per-person marginal log-likelihood (Bock–Aitkin), for the M-dimensional model:
$$
\ell(\Psi; x_i) = \log\!\int \prod_{j=1}^{J} f(x_{ij}\mid \Psi, \theta)\, g(\theta;\Psi)\, d\theta,
\qquad
\ell(\Psi; x_1,\dots,x_N) = \sum_{i=1}^N \ell(\Psi; x_i).
$$
Per-person score (`P` = number of params): `s(Psi; x_i) = (∂ℓ/∂Psi_1, …, ∂ℓ/∂Psi_P)`, with `Σ_i s(Psi_hat; x_i) = 0` at the MLE.

**Test of distinguishability.** Population variance of casewise log-likelihood ratios:
$$
\omega_*^2 = \operatorname{Var}\!\left[\log^2 \frac{f_A(x_i;\Psi_A^*)}{f_B(x_i;\Psi_B^*)}\right],
$$
estimated by
$$
\hat\omega^2 = \frac1N\sum_{i=1}^N\!\left[\log\frac{f_A(x_i;\hat\Psi_A)}{f_B(x_i;\hat\Psi_B)}\right]^2 - \left[\frac1N\sum_{i=1}^N\log\frac{f_A(x_i;\hat\Psi_A)}{f_B(x_i;\hat\Psi_B)}\right]^2.
\tag{12}
$$
Hypotheses `H0: ω²_* = 0` (indistinguishable) vs `H1: ω²_* > 0`. Under H0, `N·ω̂²` follows a **weighted sum of χ²**, weights = squared eigenvalues of a matrix built from both models' scores and information matrices (Merkle et al. 2016 appendix); tail computed via `CompQuadForm`.

**Non-nested goodness-of-fit.** Compare mean casewise log-likelihoods: `H0: E[ℓ(Ψ_A*;x_i)] = E[ℓ(Ψ_B*;x_i)]`. Statistic
$$
LR_{AB} = N^{-1/2}\sum_{i=1}^N \log\frac{f_A(x_i;\hat\Psi_A)}{f_B(x_i;\hat\Psi_B)}
\;\xrightarrow{d}\; N(0, \omega_*^2)\ \text{(when distinguishable)}.
\tag{15}
$$
`nonnest2` rescales to a standard-normal z:
$$
z = \frac{\sum_i \log\big(f_A(x_i;\hat\Psi_A)/f_B(x_i;\hat\Psi_B)\big)}{\sqrt{N}\,\hat\omega}.
$$

**Nested case.** (12) and (15) test the same hypothesis. If one assumes Model A correctly specified, `N·ω̂²` and `2·N^{1/2}·LR_{AB}` converge to ordinary χ²; if not, to weighted sums of χ² with the same eigenvalue weights.

The paper obtains information matrices via the **Oakes-identity observed information** of Chalmers (2018a) / Pritikin (2017) — i.e. Paper 2 is a prerequisite building block.

### Implementation plan

E-step / likelihood quantities needed:
- **Casewise marginal log-likelihood** under each fitted model, `ℓ(Ψ; x_i)` — the engine already forms per-person marginal likelihoods on the quadrature grid (used for the global loglik and for S-X²); expose the per-person vector for both models.
- **Casewise score vectors** `s(Ψ; x_i)` — new: the per-person gradient of the marginal loglik. Derivable by the Fisher/Louis identity as the posterior-weighted complete-data score, `s(Ψ;x_i) = E_{θ|x_i}[∂ log f(x_i,θ;Ψ)/∂Ψ]`, reusing the same posterior weights the M-step already computes (no new integration rule).
- **Observed information** per model — from Paper 2's Oakes estimator.

New computation:
1. Both models fit on the **same data with the same quadrature**; align/pad score and parameter vectors.
2. `d_i = log f_A(x_i;Ψ̂_A) − log f_B(x_i;Ψ̂_B)`; then `ω̂²` (12), `z` (15) — trivial once `d_i` exists.
3. Distinguishability weights: build the block matrix `W` from `A_m = −(1/N)·(observed information)_m` and `B_m = (1/N)·Σ_i s_m(x_i)s_m(x_i)ᵀ` for `m ∈ {A,B}` plus the cross block `B_{AB} = (1/N)·Σ_i s_A(x_i)s_B(x_i)ᵀ`, take its eigenvalues, and evaluate the weighted-χ² tail (Davies/Imhof; a small self-contained routine, no external dependency needed).

Computational cost: casewise loglik is already computed; casewise scores add one posterior-weighted gradient pass (`O(N · P_item)`), cheap. The eigenvalue step is `O((P_A+P_B)³)` once — negligible. Weighted-χ² tail is a 1-D numeric integral.

Minimal correct test: fit a 1PL (Rasch, slopes fixed) and a 2PL to data simulated from the 2PL; assert the **nested** Vuong z favors the 2PL (`z` significant in the 2PL direction) and that the distinguishability test rejects `H0` (`N·ω̂² ` tail p small). Sanity check: `Σ_i s(Ψ̂;x_i) ≈ 0` at each MLE (gradient-zero identity).

### Not implementable / out of scope

- **Weighted-χ² tail**: needs a self-contained Davies/Imhof routine (the repo bans SciPy); a modest but real new numeric primitive — flag as the one non-trivial dependency to build.
- Graded-response vs GPCM comparisons in the paper require **polytomous** models — not in fast-mlsirm yet; only 2PL-vs-Rasch / dimension-count comparisons are testable today.
- Requires Paper 2's Oakes information as a prerequisite; do Paper 2 first.

---

## Paper 3 — Kang, Cohen & Sung (2009): Model selection indices for polytomous items

**Full citation.** Taehoon Kang, Allan S. Cohen & Hyun-Jung Sung (2009). *Model Selection Indices for Polytomous Items.* **Applied Psychological Measurement** 33(7): 499–518. DOI: 10.1177/0146621608327800.

**What it is.** Compares four indices — AIC, BIC (both MMLE), DIC and cross-validation log-likelihood CVLL (both MCMC/Bayesian) — for choosing among four polytomous IRT models (RSM, PCM, GPCM, GRM). Verdict: **BIC is the most accurate and consistent** (98% correct over 1,600 data sets, 100% for PCM/RSM); AIC nearly ties BIC but over-selects the more complex model as `N` grows; DIC and CVLL need large `N` *and* many categories to reliably pick the GRM over the GPCM.

### Exact definitions

Deviance `= −2·log(marginal maximum likelihood)`, `p` = number of estimated parameters, `N` = sample size.

$$
\mathrm{AIC} = -2\log L(\hat\theta) + 2p
$$
$$
\mathrm{BIC} = -2\log L(\hat\theta) + p\log N
$$

DIC (Spiegelhalter et al. 2002), eq (4), with `D(y)` the deviance:
$$
\mathrm{DIC} = \overline{D(y)} + p_D = D(\bar y) + 2p_D,
\qquad
p_D = \overline{D(y)} - D(\bar y),
$$
where `\overline{D(y)}` is the posterior mean deviance and `D(\bar y)` the deviance at the posterior-mean parameters. Smallest DIC wins.

CVLL (Geisser–Eddy / Gelfand–Dey), eq (5): split into calibration `Y_cal` and cross-validation `Y_cv` samples; use the `Y_cal` posterior as the prior:
$$
P(Y_{cv}\mid \text{Model}) = \int P(Y_{cv}\mid \theta, Y_{cal}, \text{Model})\, f_\theta(\theta \mid Y_{cal}, \text{Model})\, d\theta,
\qquad
\mathrm{CVLL} = \log P(Y_{cv}\mid \text{Model}).
$$
**Largest** CVLL wins (opposite sign convention to AIC/BIC/DIC).

### Recommendation (explicit)

**BIC.** Most accurate and consistent across all 32 conditions; least likely to over-parameterize; works even at `N=500`. AIC ≈ BIC except it tends to pick the more complex model at large `N` (confirmed in their real-data example where AIC alone chose GPCM over the more parsimonious PCM). DIC/CVLL are only competitive at `N=1000` with 5-category items.

### Implementation plan

**AIC and BIC are already implemented** in `python/fast_mlsirm/diagnostics.py`:
```
aic = 2.0 * n_parameters - 2.0 * loglik
bic = np.log(n_observed) * n_parameters - 2.0 * loglik
```
So the *actionable* deliverable from this paper for an MMLE engine is essentially: **surface BIC as the recommended default index, and confirm `n_parameters` and `n_observed` are counted correctly** for each population structure. E-step quantity needed: the marginal log-likelihood — already the EM convergence criterion.

`n_parameters` accounting to verify (the only real work): per binary item `alpha, b, zeta(K)` plus global `tau`, plus free population moments — multigroup adds `(mu_gd, sigma_gd)` for non-reference groups; multilevel adds `sigma_u`; FIPC freezes anchored items (do not count them). `N` for BIC should be the number of persons (response vectors), not the number of observed cells — verify against `n_observed` in `diagnostics.py`.

Computational cost: zero beyond the existing fit.

Minimal correct test: simulate from a 1PL, fit 1PL and 2PL, assert **BIC(1PL) < BIC(2PL)** (parsimony favored) while the raw loglik of 2PL ≥ 1PL; and assert `AIC = 2p − 2·loglik`, `BIC = log N·p − 2·loglik` exactly for a hand-checked `p`, `N`.

### Not implementable / out of scope

- **DIC and CVLL as defined here are Bayesian/MCMC** — they need a posterior sample (`\overline{D(y)}`, posterior-mean parameters, and the `Y_cal`-posterior-as-prior integral). fast-mlsirm is MMLE with no sampler → **out of scope**. Note the engine already has an MLE-based held-out log-likelihood in `diagnostics.py` (`heldout_loglik`), which is the frequentist analogue of CVLL and serves the same "predict a replicate sample" goal without a sampler; document that as the substitute rather than porting Bayesian CVLL.
- All four candidate models here (RSM/PCM/GPCM/GRM) are **polytomous** — the *index formulas* are model-agnostic and apply to binary models unchanged, but the paper's specific model-selection scenarios are not reproducible until polytomous items land.

---

## Paper 4 — Svetina & Levy (2014): A framework for dimensionality assessment for MIRT

**Full citation.** Dubravka Svetina & Roy Levy (2014). *A Framework for Dimensionality Assessment for Multidimensional Item Response Models.* **Educational Assessment** 19: 35–57. DOI: 10.1080/10627197.2014.869450.

**What it is.** Not a new method — a **taxonomy/framework** that classifies existing dimensionality-assessment procedures along four axes: exploratory vs confirmatory, parametric vs nonparametric, item-response type (dichotomous / ordered polytomous), and data features (lower asymptote / missing data). It situates ~10 procedures (EFA + parallel analysis, χ² difference test, DETECT/PolyDETECT, DIMTEST/PolyDIMTEST, NOHARM `χ²_{G/D}` & ALR, WRMR, RMSR change, local-dependence indices Q3 / model-based covariance / X²·G², and the GDDM via PPMC) and illustrates each on NAEP Science data.

### The concrete procedures and the formulas they use

Local independence (the unifying criterion), eq (3):
$$
P(X\mid\theta,\omega) = \prod_{j=1}^{J} P(X_j\mid\theta,\omega_j).
$$
Weak (pairwise) local independence, eq (4):
$$
E_\theta\!\big[\operatorname{Cov}(X_j, X_{j'}\mid\theta,\omega)\big]
= E_\theta\big[(X_j - E(X_j\mid\theta,\omega_j))(X_{j'} - E(X_{j'}\mid\theta,\omega_{j'}))\big] = 0.
$$

Compensatory MIRT (dichotomous), eq (1): `P(X_ij = 1) = c_j + (1 − c_j) F(a_jᵀθ_i + d_j)`.

Item-pair **local-dependence indices** (the implementable core):
- **Yen's Q3**: residual `d_ij = X_ij − E(X_ij | θ̂_i)`; `Q3_{jj'} = corr(d_{·j}, d_{·j'})`. Flag `|Q3| > 0.20`.
- **Model-based covariance** (Reckase 1997): `Cov_model(X_j, X_{j'})` using model-implied expected values.
- **GDDM** (Levy & Svetina 2011): test-level average of absolute model-based covariance over item pairs,
$$
\mathrm{GDDM} = \frac{2}{J(J-1)} \sum_{j<j'} \big|\,\widehat{\operatorname{Cov}}(X_j, X_{j'})\,\big|.
$$
- **X² / G²** contingency-table statistics on bivariate response patterns vs model-implied expectations.

Dimensionality-count procedures:
- **Parallel analysis** (Horn 1965): factor the tetrachoric (binary) / polychoric (poly) correlation matrix; retain factors whose observed eigenvalue exceeds the mean (or 95th percentile) of eigenvalues from many random-data matrices of the same `(N, J)`.
- **χ² difference test** for `M` vs `M+1` factors, referred to χ² with `Δdf`; Schilling–Bock caution: require `χ² > 2·df` (empirical Type-I inflation).
- **NOHARM `χ²_{G/D}`** and **ALR** — from residual correlations / bivariate LR of a fitted NOHARM model; used in a sequential fit.
- **RMSR change** (Tate 2003): add factors until RMSR reduction < 10%.
- **DETECT / PolyDETECT**: nonparametric; partition items to maximize within-cluster-positive / between-cluster-negative conditional covariance; `D_ref` cutoffs (<.20 unidim; .20–.39 weak; .40–.79 moderate; >.80 strong), plus IDN and ratio R (≥.80 ⇒ simple structure).
- **DIMTEST / PolyDIMTEST**: Stout's `T` statistic aggregating conditional covariance of an assessment subtest (AT) conditional on a partitioning subtest (PT); asymptotically normal under essential unidimensionality.
- **PPMC** (posterior predictive model checking): posterior-predictive p-value = tail area of the reference distribution of a discrepancy (e.g. GDDM) — Bayesian.

### Implementation plan

fast-mlsirm supports simple-structure MIRT (`theta in R^D`), so *confirmatory* dimensionality checks on a specified structure are the natural fit. Directly / adaptably implementable:

- **Q3 and model-based covariance / GDDM** — the highest-value, lowest-cost items. E-step quantities needed: EAP trait scores `θ̂_i` (or full posterior) and model-implied `E(X_ij | θ)`, both already available (the engine computes posterior expectations for S-X² and scoring). New computation: residuals `d_ij`, their `J×J` correlation matrix (Q3), and the pairwise model-based covariance average (GDDM). Cost `O(N·J²)`, one pass. This is a small addition to `diagnostics.py`.
- **χ² difference test** for nested dimensional structures (`D` vs `D+1` traits) — fit both, `Δdeviance ~ χ²(Δdf)`, apply the `χ² > 2·df` guard. Reuses the marginal loglik; near-zero cost. (Overlaps with Paper 1's nested test — Vuong is the more robust version.)
- **Parallel analysis** on the tetrachoric correlation matrix — standalone, doesn't even need a fit; a compact numeric routine (needs a tetrachoric-correlation estimator + eigen-decomposition + random-data resampling). Moderate new code.

Minimal correct test: simulate a 2-dimensional simple-structure data set with one deliberately cross-loading item; assert Q3 flags that item pair (`|Q3| > 0.2`) and GDDM is larger than for a clean unidimensional fit. For the χ² difference test: simulate 1-D data, fit 1-D and 2-D, assert the difference test does **not** reject with the `2·df` guard.

### Not implementable / out of scope

- **DETECT, DIMTEST, NOHARM `χ²_{G/D}`, ALR, WRMR** are **separate software/algorithms** (raw-score conditional-covariance partitioning, Stout's `T`, NOHARM's least-squares residual machinery). Porting any is a project unto itself and mostly duplicates what a confirmatory Q3/GDDM already tells a simple-structure engine → **document-only**, not worth building.
- **PPMC / posterior-predictive p-values are Bayesian** (need an MCMC posterior) → out of scope for MMLE. The frequentist substitute is to reference GDDM/Q3 against a parametric-bootstrap reference distribution instead of a posterior-predictive one.
- **Polytomous / lower-asymptote (3PL guessing)** branches of the framework don't apply — fast-mlsirm is binary, no `c_j`.
- The paper is fundamentally a **review/taxonomy**; its deliverable to fast-mlsirm is the *classification* (use it to justify shipping confirmatory Q3/GDDM/χ²-difference and to document why DETECT/DIMTEST are out of scope), not an algorithm.

---

## Paper 5 — Sinharay & Lu (2008): Correlation between item parameters and item-fit statistics

**Full citation.** Sandip Sinharay & Ying Lu (2008). *A Further Look at the Correlation Between Item Parameters and Item Fit Statistics.* **Journal of Educational Measurement** 45(1): 1–15.

**What it is.** Revisits Dodeen's (2004) worrying claim that item-fit statistics correlate with item parameters (so highly-discriminating items falsely look misfitting). Sinharay & Lu show that Dodeen's result is an artifact of a **bad fit statistic** (`χ²_G`, which uses point-θ groupings and has grossly inflated Type-I error). With statistics that have correct Type-I error — especially **S-X²** (Orlando & Thissen 2000) — there is **no** spurious correlation with item parameters. Recommendation: **use S-X² (or S-G²)**.

### Exact statistics

`O_j`, `E_j` = observed / expected proportion correct in group `j`; `N_j` = group size; `n` = number of groups.

`χ²_G` / G²-like (Mislevy–Bock), groups on **proficiency θ**, eq (1) — the *bad* one:
$$
\chi^2_G = 2\sum_{j=1}^n N_j\!\left[O_j\log\frac{O_j}{E_j} + (1-O_j)\log\frac{1-O_j}{1-E_j}\right].
$$
Standardized residual (Hambleton et al.), eq (2): `z_j = (O_j − E_j) / sqrt(E_j(1−E_j)/N_j)`.

**S-X²** (Orlando & Thissen 2000), groups on **summed/raw score**, eq (3) — the recommended one:
$$
S\text{-}X^2 = \sum_{j=1}^n \frac{N_j\,(O_j - E_j)^2}{E_j(1-E_j)} \;\sim\; \chi^2_{\,n-4}\ \text{(3PL)}.
$$
**S-G²**, summed-score groups, eq (4):
$$
S\text{-}G^2 = 2\sum_{j=1}^n N_j\!\left[O_j\log\frac{O_j}{E_j} + (1-O_j)\log\frac{1-O_j}{1-E_j}\right].
$$
`χ²*` / `G²*` (Stone 2000): proficiency-scale groups via posterior "pseudo-counts", rescaled to a χ² reference by a resampling procedure.

Expected proportions `E_j` in S-X² come from the **Lord–Wingersky recursion** over the summed-score distribution — exactly the machinery already in fast-mlsirm's `fitstats`.

### Recommendation (explicit)

**S-X² and S-G².** Type-I error close to nominal across sample sizes and test lengths, respectably high power, and — the paper's point — **no spurious linear relationship** with discrimination/difficulty/guessing. Avoid `χ²_G` / `z_j`: their reference distribution is not χ² (they use point-θ, ignore θ uncertainty; Chernoff–Lehmann shows a plug-in-MLE χ² is stochastically larger than χ²), so they over-flag high-discrimination items. `χ²*`/`G²*` are better than `χ²_G` but showed inflated Type-I at `N=2000` in this study.

### Implementation plan

**S-X² is already implemented** in `crates/mlsirm-core/src/fitstats.rs` and `python/fast_mlsirm/fitstats.py`, with the Lord–Wingersky recursion generalized to the `(theta, xi)` grid and a practical-significance effect size (Sinharay & Haberman 2014). So this paper is **confirmatory / document-only** for fast-mlsirm: it is the citation that *justifies the existing choice* of S-X² as the default item-fit statistic and justifies *not* implementing `χ²_G`.

Actionable follow-through (small): (1) document S-X² as the recommended item-fit index in the API/help, citing Orlando–Thissen 2000 and Sinharay–Lu 2008; (2) ensure S-G² is available too (same `O_j/E_j/N_j` inputs, log form) if not already — it is a two-line addition next to S-X². E-step quantities: the summed-score distribution (Lord–Wingersky) and per-group `E_j` — already computed.

Minimal correct test: on data simulated from the fitted model, assert S-X² Type-I behaves (across ~100 replications, the rejection rate at α=.05 is ≈ .05, not inflated), and — the paper's specific claim — assert the correlation between generating discrimination `alpha` and average S-X² across items is not significant, whereas the same correlation for a `χ²_G`-style point-θ statistic *is* inflated. A cheaper unit test: assert S-X² on model-consistent data has expected value ≈ its degrees of freedom `(n − k)`.

### Not implementable / out of scope

- `χ²_G`, `z_j`, `χ²*`/`G²*`: **deliberately not worth implementing** — the paper's whole message is that the θ-grouped point-estimate statistics have broken Type-I error. Building them would be building known-bad diagnostics. `ponytail:` skip.
- The paper uses the **3PL** (guessing `c`); fast-mlsirm is 2PL/binary with no lower asymptote, so the `n−4` df (which subtracts 3 item params + 1) becomes `n−3` for a 2PL — note the df bookkeeping difference when documenting.

---

### Cross-paper build order (recommendation)

1. **Paper 2 (Oakes)** — direct, high value, and a prerequisite for Paper 1. Ship first.
2. **Paper 3 (BIC)** — already there; just surface/verify parameter counting. Trivial.
3. **Paper 5 (S-X²)** — already there; add S-G² + docs. Trivial.
4. **Paper 4 (Q3 / GDDM / χ²-difference)** — small confirmatory additions to `diagnostics.py`; parallel analysis optional.
5. **Paper 1 (Vuong)** — adaptation; needs casewise scores + a weighted-χ² tail routine; do after Oakes lands.
