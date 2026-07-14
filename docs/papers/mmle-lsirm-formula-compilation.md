# Mathematical Foundations for an MMLE-Estimated Latent Space Item Response Model (LSIRM) and Its Extensions

**Purpose.** Reference specification for implementing, in software, a marginal-maximum-likelihood
(MMLE) estimated LSIRM together with multilevel, multigroup, and multidimensional extensions, plus
item-fit and person-fit statistics. Every model equation is given in LaTeX with parameter definitions,
identifiability constraints, and concrete estimation update equations.

**Verification legend** (see also §12):
`[V]` = equation verified verbatim against the cited primary source online in this compilation.
`[S]` = standard textbook result reproduced from memory (source cited, exact page not re-fetched).
`[~]` = partially verified (existence/description confirmed online; exact formula from memory).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `p = 1,…,N` (or `k`) | respondents / persons |
| `i = 1,…,I` (or `j`) | items |
| `Y_{pi} ∈ {0,1}` | binary item response (LSIRM-continuous uses `Y_{pi} ∈ ℝ`) |
| `θ_p` (paper writes `α_j`) | person main effect / latent trait ("ability") |
| `β_i` | item main effect ("easiness"; note sign convention `+β_i`) |
| `z_p ∈ ℝ^D` (paper: `a_j`) | latent position of respondent `p` |
| `w_i ∈ ℝ^D` (paper: `b_i`) | latent position of item `i` |
| `γ ≥ 0` | weight of the distance (interaction) term |
| `d(·,·)` | distance on the latent metric space (default Euclidean `ℓ₂`) |
| `D` | dimension of the latent space (map), `D≥1`, typically `D=2` |
| `g(θ)` | population density of the latent trait (MMLE integrates over this) |

> **Notation bridge.** The task statement uses `θ_p + β_i − γ·d(z_p,w_i)`. Jeon et al. (2021) write the
> identical model as `α_j + β_i − γ·d(a_j,b_i)`. This document uses `(θ_p, z_p, w_i)` throughout and notes
> the `(α_j, a_j, b_i)` originals where quoting the paper.

---

## 1. Base LSIRM (Jeon, Jin, Schweinberger & Baugh, 2021) `[V]`

### 1.1 Model

General interaction form (paper Eq. 2), with `g(·,·)` a real-valued function of the two positions:
$$
\operatorname{logit}\!\big(P(Y_{pi}=1\mid \theta_p,\beta_i,z_p,w_i)\big)=\theta_p+\beta_i+g(z_p,w_i).
$$

Two choices of `g`:

- **Distance effect (the LSIRM proper, recommended by the authors):**
$$
g(z_p,w_i)=-\gamma\, d(z_p,w_i),\qquad \gamma\ge 0,
$$
giving the working model
$$
\boxed{\;\operatorname{logit}\!\big(P(Y_{pi}=1)\big)=\theta_p+\beta_i-\gamma\,\lVert z_p-w_i\rVert\;}
\tag{LSIRM}
$$
`γ>0` makes the success probability **decrease** in the respondent–item distance. Distance choices
discussed: `ℓ₁` (city-block), `ℓ₂` (Euclidean, default), `ℓ∞` (maximum).

- **Multiplicative (bilinear) effect:** `g(z_p,w_i)=z_p^⊤ w_i` (inner product). Related to Hoff's (2005)
bilinear mixed-effects / additive-and-multiplicative-effects network models. Harder to interpret (the
effect is 0 whenever the vectors are orthogonal, regardless of distance), so the paper focuses on the
distance form.

**Relation to other models** (paper §2.3.2): the 2-parameter IRT model
`logit P = λ_i θ_p + β_i` and the saturated interaction model `logit P = θ_p + β_i + ε_{pi}` are alternatives;
the LSIRM is the special case `ε_{pi} = −γ d(z_p,w_i)`. The distance restriction (reflexivity, symmetry,
triangle inequality) is what makes the interaction **estimable** from a single response per (p,i) pair
and injects transitivity (nearby respondents behave similarly).

### 1.2 Priors (fully Bayesian original) `[V]`

$$
\begin{aligned}
\theta_p\mid\sigma^2 &\overset{ind}{\sim} N(0,\sigma^2), & \sigma^2>0,\\
\beta_i\mid\tau_\beta^2 &\overset{ind}{\sim} N(0,\tau_\beta^2), & \tau_\beta^2>0,\\
\log\gamma\mid\mu_\gamma,\tau_\gamma^2 &\sim N(\mu_\gamma,\tau_\gamma^2), & \mu_\gamma\in\mathbb R,\ \tau_\gamma^2>0,\\
\sigma^2\mid a_\sigma,b_\sigma &\sim \text{Inv-Gamma}(a_\sigma,b_\sigma), & a_\sigma,b_\sigma>0,\\
z_p &\overset{iid}{\sim}\mathrm{MVN}_D(\mathbf 0, I_D), & p=1,\dots,N,\\
w_i &\overset{iid}{\sim}\mathrm{MVN}_D(\mathbf 0, I_D), & i=1,\dots,I.
\end{aligned}
$$
Default hyperparameters used in the paper: `τ_β²=4, a_σ=1, b_σ=1, μ_γ=0.5, τ_γ²=1`.
A prior is placed on **positions** (not distances) because distances must satisfy the triangle inequality,
which is awkward to encode directly.

Joint posterior (paper Eq. 5):
$$
f(\theta,\beta,\gamma,Z,W\mid y)\propto
\Big[\textstyle\prod_p f(\theta_p)\Big]\Big[\prod_i f(\beta_i)\Big]f(\gamma)
\Big[\prod_p f(z_p)\Big]\Big[\prod_i f(w_i)\Big]
\prod_{p}\prod_{i}P(Y_{pi}=y_{pi}\mid\theta_p,\beta_i,\gamma,z_p,w_i).
$$

### 1.3 Estimation in the original: MCMC (Metropolis-within-Gibbs) `[V]`

Component-wise updates per iteration `t`; each block accepted with the usual MH ratio
`min{1, f(·*|rest)/f(·^{(t)}|rest)}` using symmetric (multivariate) Gaussian random-walk proposals
centered at the current value with diagonal covariance, tuned to an acceptance rate ≈ 0.3:

1. `θ_p*` (all `p`); 2. `β_i*` (all `i`); 3. `γ*`; 4. `z_p*` (all `p`); 5. `w_i*` (all `i`);
6. Gibbs draw of `σ²` from its full conditional:
$$
\sigma^2\sim \text{Inv-Gamma}\!\left(a_\sigma+\tfrac{N}{2},\; b_\sigma+\tfrac{1}{2}\textstyle\sum_{p=1}^N\theta_p^2\right).
$$
Convergence via trace plots + Gelman–Rubin `R̂`.

### 1.4 Identifiability of the latent space `[V]`

The log-odds depends on positions only through **distances**, which are invariant to **translation,
rotation, and reflection** of the whole configuration; hence the likelihood is invariant under these
transformations (the same non-identifiability as in latent space network models, Hoff, Raftery &
Handcock, 2002). Resolution: **post-process the MCMC/optimization output with Procrustes matching**
(Gower, 1975) to a reference configuration; interpret only **relative** distances. Additional practical
pins: the `MVN_D(0, I_D)` prior centers the map at the origin (removes translation); for a point estimate,
also fix scale. The multiplicative/inner-product variant has **only rotational** invariance
(`z^⊤w = (Rz)^⊤(Rw)` for orthogonal `R`).

> **Implementation note.** For an MMLE/point-estimate pipeline, resolve invariance by: (i) mean-centering
> `Z` and `W` each iteration (translation); (ii) Procrustes-rotating the current `W` to a fixed reference
> `W₀` (rotation+reflection); (iii) fixing `γ>0` scale or standardizing position variance. Anchoring a few
> items' positions is an alternative that also enables cross-group comparability (see §6).

### 1.5 Model selection (`γ=0` Rasch vs. `γ>0` LSIRM)

Compare the Rasch/1PL nested model (`γ=0`) against LSIRM. The original uses Bayesian comparison; the R
package uses **BIC** and maximum log-posterior (§2.4). A spike-and-slab mixture prior on `γ` (mass near 0
vs. spread over positives) yields a built-in test of whether an interaction map is needed.

---

## 2. LSIRM variants (Go, Kim, Park, Park, Jeon & Jin — `lsirm12pl`) `[V]`

### 2.1 1PL LSIRM (binary) — as §1, package Eq. (2)
$$
\operatorname{logit}\!\big(P(Y_{pi}=1\mid\theta_p,\beta_i,\gamma,z_p,w_i)\big)=\theta_p+\beta_i-\gamma\,\lVert z_p-w_i\rVert,\qquad \theta_p\sim N(0,\sigma^2).
$$

### 2.2 2PL LSIRM (binary)
$$
\operatorname{logit}\!\big(P(Y_{pi}=1)\big)=\alpha_i\,\theta_p+\beta_i-\gamma\,\lVert z_p-w_i\rVert,\qquad \theta_p\sim N(0,\sigma^2).
$$
`α_i` = item discrimination. **Identification of slopes:** fix one slope, `α_1=1`.
**Prior:** `log α_i ∼ N(μ_α, τ_α²)` (log-normal, keeps `α_i>0`); package defaults `μ_α=0.5, τ_α=1`.
All other priors as in §1.2.

### 2.3 Continuous (Gaussian) LSIRM — identity link
$$
\begin{aligned}
\text{1PL:}\quad Y_{pi}&=\theta_p+\beta_i-\gamma\,\lVert z_p-w_i\rVert+\epsilon_{pi},\\
\text{2PL:}\quad Y_{pi}&=\alpha_i\theta_p+\beta_i-\gamma\,\lVert z_p-w_i\rVert+\epsilon_{pi},
\end{aligned}
\qquad \epsilon_{pi}\sim N(0,\sigma_\epsilon^2),\ \theta_p\sim N(0,\sigma^2).
$$
Likelihood is a product of normals `∏_p ∏_i N(Y_{pi};\,\mu_{pi},\sigma_\epsilon^2)` with mean `μ_{pi}` the
linear predictor. Extra prior: `σ_ε² ∼ Inv-Gamma(a_{σε}, b_{σε})`. Two distinct variance components:
`σ²` = prior variance of `θ_p`; `σ_ε²` = residual variance.

### 2.4 Estimation & fit in the package
Fully Bayesian **Metropolis-Hastings-within-Gibbs** (Chib & Greenberg, 1995) for all of
`θ,β,γ,Z,W` (plus `α`, `σ_ε²`). MAR missingness handled by data augmentation (Tanner & Wong, 1987).
Identifiability by **Procrustes** post-processing (Gower, 1975). Reported diagnostics: **BIC**, max
log-posterior, posterior-predictive item-mean plots + ROC/AUC (binary), trace/ACF/Gelman–Rubin–Brooks.
No MML/EM/variational in the package — motivating §3–§4 below.

---

## 3. MMLE / EM formulation (the frequentist estimation target)

LSIRM has latent quantities **per person** (`θ_p`, `z_p ∈ ℝ^D`) and **per item** (`w_i ∈ ℝ^D`), plus
structural parameters `ξ = (β, γ, σ², [α], [σ_ε²])`. Two coherent frequentist framings:

### 3.A Random-effects / marginal likelihood (persons integrated out) `[S]`

Treat person latents `(θ_p, z_p)` as random effects with densities `θ_p∼N(0,σ²)`, `z_p∼MVN_D(0,I_D)`;
treat item positions `w_i`, `β_i`, `γ` as **structural parameters** to estimate. The marginal likelihood is
$$
L(\xi, W;\,y)=\prod_{p=1}^{N}\ \int_{\mathbb R}\!\int_{\mathbb R^{D}}
\ \prod_{i=1}^{I} P_{pi}^{\,y_{pi}}\,(1-P_{pi})^{\,1-y_{pi}}\ \phi(\theta_p;\sigma^2)\,\phi_D(z_p)\ dz_p\,d\theta_p,
$$
with `P_{pi}=\operatorname{logit}^{-1}(\theta_p+\beta_i-\gamma\lVert z_p-w_i\rVert)`. This is the LSIRM analogue
of Bock & Aitkin (1981). The item positions `w_i` are *not* integrated out here — they are the map we want.
(One may symmetrically put `w_i` as random effects and integrate them out too; then `Z` is estimated, or
one alternates — see §3.C MH-RM, which handles both cleanly.)

#### 3.A.1 Bock–Aitkin EM for the trait margin (classical IRT baseline) `[~]`
For the **ability-only** margin (fixing `γ`, `Z` momentarily, or for a plain 2PL calibration), the classic
EM with Gauss–Hermite quadrature applies. Approximate `∫ h(θ)g(θ)dθ ≈ Σ_{q=1}^{Q} h(X_q)A_q` at nodes
`X_q` with weights `A_q`.

**E-step.** Posterior weight of node `q` for person `p`:
$$
P(X_q\mid y_p)=\frac{L_p(X_q)\,A_q}{\sum_{q'=1}^{Q}L_p(X_{q'})\,A_{q'}},\qquad
L_p(X_q)=\prod_i P_i(X_q)^{y_{pi}}\big(1-P_i(X_q)\big)^{1-y_{pi}}.
$$
Expected counts (artificial data):
$$
\bar N_q=\sum_{p=1}^{N}P(X_q\mid y_p),\qquad
\bar r_{iq}=\sum_{p=1}^{N}y_{pi}\,P(X_q\mid y_p).
$$

**M-step.** For each item `i`, maximize the expected complete-data log-likelihood
$$
\sum_{q=1}^{Q}\Big[\bar r_{iq}\log P_i(X_q)+(\bar N_q-\bar r_{iq})\log\big(1-P_i(X_q)\big)\Big]
$$
i.e. a weighted binomial fit. For the 2PL `P_i(θ)=\operatorname{logit}^{-1}(α_i θ+β_i)` the likelihood
equations are
$$
\sum_{q}\big(\bar r_{iq}-\bar N_q P_i(X_q)\big)=0,\qquad
\sum_{q}\big(\bar r_{iq}-\bar N_q P_i(X_q)\big)X_q=0,
$$
solved by Newton–Raphson / Fisher scoring. Iterate E/M to convergence. The population `σ²` (or a free mean)
is updated from the moments of the posterior `P(X_q|y_p)`.

#### 3.A.2 Why plain quadrature fails for the full LSIRM, and what to do `[S]`
The per-person latent is `(1+D)`-dimensional; a `Q`-point grid needs `Q^{1+D}` nodes (curse of
dimensionality), infeasible for `D≥2`. Practical E-steps:

- **Monte-Carlo / importance-sampling E-step (MCEM):** draw `m` samples `(θ_p^{(s)}, z_p^{(s)})` from (an
  approximation to) the posterior `f(θ_p,z_p\mid y_p,\xi,W)` and replace the integral by the sample average.
  Expected complete-data log-likelihood gradient wrt structural params:
  `∇_ξ Q ≈ (1/m) Σ_s ∇_ξ log f(y_p,θ_p^{(s)},z_p^{(s)};ξ,W)`.
- **Adaptive Gauss–Hermite** (Laplace-centered nodes per person) — viable for small `D`.
- **Stochastic EM / MH-RM (§3.C)** — the recommended route for LSIRM.

### 3.B Joint maximum likelihood (JML) — Hoff–Raftery–Handcock lineage `[~]`

Treat **all** `θ_p, z_p, w_i, β_i, γ` as fixed parameters and maximize the joint log-likelihood
$$
\ell(\Xi)=\sum_{p=1}^{N}\sum_{i=1}^{I}\Big[y_{pi}\log P_{pi}+(1-y_{pi})\log(1-P_{pi})\Big],
$$
by block coordinate ascent (gradient steps alternating persons ↔ items). Hoff, Raftery & Handcock (2002)
introduced this exact idea for **latent space network models**: obtain MLE latent positions (they used
distances from a logistic regression + MDS start), then refine — and it transfers directly to LSIRM's
distance model. Gradients (Euclidean distance `d_{pi}=\lVert z_p-w_i\rVert`, unit vector
`u_{pi}=(z_p-w_i)/d_{pi}`):
$$
\frac{\partial\ell}{\partial\theta_p}=\sum_i (y_{pi}-P_{pi}),\quad
\frac{\partial\ell}{\partial\beta_i}=\sum_p (y_{pi}-P_{pi}),\quad
\frac{\partial\ell}{\partial\gamma}=-\sum_{p,i}(y_{pi}-P_{pi})\,d_{pi},
$$
$$
\frac{\partial\ell}{\partial z_p}=-\gamma\sum_i (y_{pi}-P_{pi})\,u_{pi},\qquad
\frac{\partial\ell}{\partial w_i}=+\gamma\sum_p (y_{pi}-P_{pi})\,u_{pi}.
$$
**Caveats:** JML latent positions are unidentified up to translation/rotation/reflection (re-Procrustes
each iteration; center `Z,W`); and JML suffers the **incidental-parameters (Neyman–Scott) problem** — with
person and item latents both growing, estimates of structural parameters can be inconsistent. Use JML for
a fast warm start, then hand off to the marginal estimator (§3.A/§3.C) for consistent structural estimates.
A ridge/prior penalty (equivalently the `MVN_D(0,I_D)` prior as an `ℓ₂` penalty on positions) regularizes
the otherwise flat directions.

### 3.C MH-RM (Cai, 2010) — recommended MML estimator for LSIRM `[V for algorithm, ~ for LSIRM specialization]`

Metropolis–Hastings Robbins–Monro is stochastic-approximation EM built for exactly this regime (many
latents, high dimension). Let complete data be `(y, φ)` with latent `φ=({θ_p,z_p}_p, [ {w_i}_i ])` and
structural `ξ`. Uses **Fisher's identity** `∇_ξ log L(ξ) = E_φ[ ∇_ξ log f(y,φ;ξ) \mid y,ξ ]`.

Iteration `t`:
1. **Imputation (MH):** draw `φ^{(t)}` with a few Metropolis–Hastings steps from `f(φ\mid y,\xi^{(t-1)})`
   (random-walk proposals on `θ_p, z_p, w_i`).
2. **Approximation:** form the complete-data score (ascent direction) and (optionally) an information
   estimate at the imputed data,
   $$
   s^{(t)}=\nabla_\xi \log f\big(y,\varphi^{(t)};\xi^{(t-1)}\big),\qquad
   H^{(t)}=\text{recursive estimate of }-\nabla^2\ \text{(or empirical info)}.
   $$
3. **Robbins–Monro update:**
   $$
   \xi^{(t)}=\xi^{(t-1)}+\varepsilon_t\,\big(H^{(t)}\big)^{-1}s^{(t)},\qquad
   \sum_t \varepsilon_t=\infty,\ \ \sum_t \varepsilon_t^2<\infty\ (\text{e.g. }\varepsilon_t=1/t).
   $$
The estimate sequence converges w.p.1 to the MML solution. Standard errors come from the recursive
information accumulation (Louis's identity). Because MH-RM only needs to *sample* `φ` (never integrate),
it sidesteps the `Q^{1+D}` quadrature blow-up — the practical reason to prefer it for LSIRM. (Reference
implementation for multidimensional IRT: `mirt::mirt(..., method = "MHRM")`.)

### 3.D Variational / importance-sampling MML `[~]`

Mean-field or Gaussian variational inference maximizes the ELBO
$$
\log p(y)\ \ge\ \mathcal L(q)=\mathbb E_{q}\big[\log p(y,\varphi,\xi)\big]-\mathbb E_q[\log q(\varphi)],
$$
with `q(φ)=∏_p N(θ_p;m_p,s_p^2)\,N_D(z_p;μ_p,Σ_p)\,∏_i N_D(w_i;ν_i,Ω_i)`. For the latent-space *network*
model, Gaussian VI (Salter-Townshend & Murphy, 2013) gives fast, scalable position estimates and transfers
to LSIRM's distance likelihood via a local (delta / quadratic) bound on `log σ(·)` or a Pólya–Gamma
augmentation for the logistic term. Variational IRT (Wu et al., 2020; Natesan-style SVI) demonstrates the
same idea for the trait margin. Treat VI as a fast approximate MML; expect variances to be under-estimated.

---

## 4. Multilevel (hierarchical) extension

### 4.1 HLSIRM (Park, Shin, Jeon, Kim & Jin, 2026) `[V]`

Students `i` nested in schools `k`, items `j`. **Inner-product** interaction with a stochastic error
(no `γ`; follows Hoff's additive-and-multiplicative-effects models):
$$
\operatorname{logit}\!\big(P(y_{ij(k)}=1)\big)=\alpha_{i(k)}+\beta_j+z_{i(k)}^{\top}w_j+\varepsilon_{ij(k)},
\qquad \varepsilon_{ij(k)}\overset{iid}{\sim}N(0,1). \tag{HLSIRM}
$$
Matrix form for school `k` (`Θ^{(k)}` the `n_k×p` logit matrix):
`logit(Θ^{(k)}) = α_{(k)} 1_p^⊤ + 1_{n_k} β^⊤ + Z^{(k)} W^⊤ + E^{(k)}`.

**Multilevel structure (random effects at the school level):**
$$
\alpha_{i(k)}\mid\alpha_{(k)},\sigma^2_{(k)}\sim N(\alpha_{(k)},\sigma^2_{(k)}),\qquad
z_{i(k)}\mid z_{(k)},\Psi_z\sim \mathrm{MVN}(z_{(k)},\Psi_z),
$$
i.e. the decomposition `θ_{pg}=μ_g+ε_{pg}` is realized here as a student intercept scattered around its
**school-level mean** `α_{(k)}`, and a student position scattered around its **school-level position**
`z_{(k)}`. `σ²_{(k)}` is a **school-specific within-school variance component**; `Ψ_z` a shared
within-school position covariance.

**Key design choice — one shared map.** Item parameters `β_j, w_j` are **common across schools**
(no hierarchy on items) → measurement invariance → schools are directly comparable inside a **single unified
interaction map**; each school has its own `(α_{(k)}, z_{(k)})` within it. (This differs from fitting
separate per-group models and stitching them together.)

**Priors / hyperpriors:**
$$
\begin{aligned}
\alpha_{(k)}\mid\sigma_\alpha^2&\sim N(\alpha_0,\sigma_\alpha^2), &
z_{(k)}\mid\Psi_z&\sim \mathrm{MVN}(z_0,\Psi_z/\kappa_0),\\
\sigma^2_{(k)}&\sim \text{Inv-Gamma}(a_\sigma,b_\sigma), &
\Psi_z&\sim \text{Inv-Wishart}(S_z,\nu_z),\\
\beta_j&\sim N(\beta_0,\tau^2), &
w_j\mid\Psi_w&\sim \mathrm{MVN}(w_0,\Psi_w),\quad \Psi_w\sim \text{Inv-Wishart}(S_w,\nu_w).
\end{aligned}
$$
Values used: `α_0=β_0=0`, `z_0=w_0=0`, `σ_α=τ=2.5` (fixed for identifiability), `ν_z=ν_w=D+1`,
`S_z=S_w=2·I`, `κ_0=1`, `a_σ=b_σ=1`, error precision fixed (`1/φ=1`, confounded with parameter scale).
**Interaction-adjusted summaries** (Kang & Jeon, 2024):
`α̃_{(k)}=α_{(k)}+\tfrac1p Σ_j z_{(k)}^⊤ w_j`, `β̃_j=β_j+\tfrac1K Σ_k z_{(k)}^⊤ w_j`.

**Estimation:** fully Bayesian MCMC; a **joint per-school** Metropolis–Hastings acceptance for the coupled
block `(α_{(k)}, α_{i(k)}, β_j, z_{(k)}, z_{i(k)}, w_j)`, with conjugate Gibbs draws for the Inv-Gamma /
Inv-Wishart variance/covariance components. **Identifiability:** inner-product form has only rotational
invariance; resolved by **Procrustes** alignment of *all* positions to a reference; cross-school
comparability comes from the shared item parameters (no per-school anchoring needed). **Checking:**
posterior-predictive replication + classification metrics (AUC, F1).

> **MMLE version of the multilevel model.** To estimate HLSIRM (or a distance-based multilevel LSIRM) by
> marginal likelihood, integrate out **both** student-level latents `(α_{i(k)}, z_{i(k)})` *and* the
> school-level latents `(α_{(k)}, z_{(k)})`, keeping `β_j, w_j` (and variance components
> `σ²_{(k)}, Ψ_z, σ_α², τ²`) as structural parameters. The nested integral factorizes over schools, so an
> MH-RM E-step samples student latents given school latents, then school latents given the rest — a natural
> two-level Gibbs imputation inside the Robbins–Monro update (§3.C). Variance components update from the
> usual random-effects EM moment equations, e.g. `σ_α² ← (1/K)Σ_k E[(α_{(k)}-α_0)²\mid y]`.

### 4.2 Generic multilevel IRT (Fox & Glas, 2001; Fox, 2010) `[S]`

A two-level model with a **measurement** level and a **structural** level. Level-1 (e.g. normal-ogive /
2PL): `P(Y_{pjk}=1)=Φ(a_i θ_{pj}-b_i)` for person `p` in group `j`. Level-2 (person abilities as outcomes):
$$
\theta_{pj}=x_{pj}^\top\beta + u_{0j}+e_{pj},\qquad u_{0j}\sim N(0,\tau^2),\ e_{pj}\sim N(0,\sigma^2),
$$
so a random-intercept model gives `θ_{pj}=γ_{00}+u_{0j}+e_{pj}`, `Var(θ)=τ^2+σ^2`, with intraclass
correlation `ρ=τ^2/(τ^2+σ^2)`. Estimated by Gibbs sampling (Fox & Glas) or MML with a nested random-effects
integral. This is the template the multilevel LSIRM specializes by adding the latent-position layer.

---

## 5. Multigroup extension

### 5.1 Bock & Zimowski (1997) multiple-group IRT `[S]`

Groups `g=1,…,G`; person `p` in group `g` has trait `θ_{pg}`. Item parameters are **common** (anchored)
across groups; group populations differ in mean/variance:
$$
P(Y_{pi}=1\mid\theta_{pg})=c_i+(1-c_i)\,\operatorname{logit}^{-1}\!\big(a_i(\theta_{pg}-b_i)\big),
\qquad \theta_{pg}\sim N(\mu_g,\sigma_g^2).
$$
**Identification:** fix one reference group `μ_1=0, σ_1^2=1` (or impose `Σ_g μ_g=0`); estimate `(μ_g,σ_g^2)`
for the others. Marginal likelihood sums the Bock–Aitkin margin (§3.A.1) group-by-group with
group-specific quadrature weights `A_{q}^{(g)}` from `N(μ_g,σ_g^2)`:
$$
L=\prod_{g}\prod_{p\in g}\ \sum_{q} A_q^{(g)}\prod_i P_i(X_q)^{y_{pi}}(1-P_i(X_q))^{1-y_{pi}}.
$$
**DIF framing / measurement invariance:** designate **anchor** items with group-invariant parameters;
allow **studied** items' `(a_i,b_i)` to differ across groups. A likelihood-ratio / Wald test on the
group-specific vs. common item parameters is the DIF test; full invariance ⇒ all items anchored.

### 5.2 Multigroup LSIRM (construction) `[~]`

No dedicated multigroup-LSIRM paper was located online; the natural specification mirrors §5.1 and the
HLSIRM invariance logic:

- **Shared map + group trait distributions:** common `β_i, w_i, γ`; group traits `θ_{pg}∼N(μ_g,σ_g²)`
  and group positions `z_{pg}∼MVN_D(m_g,Σ_g)` with a fixed reference group `μ_1=0,σ_1^2=1,m_1=0,Σ_1=I_D`.
  Enables comparing group **latent-space centroids** on one map (this is exactly what HLSIRM does with
  schools as the grouping).
- **Group-specific item positions (interaction DIF):** let `w_i^{(g)}` differ across groups for studied
  items while anchor items keep a common `w_i`; a large `γ`-weighted shift `\lVert w_i^{(g)}-w_i^{(g')}\rVert`
  flags an item whose respondent–item interaction is group-dependent — the LSIRM analogue of DIF.

**Cross-group identifiability:** the invariance (translation/rotation/reflection) must be resolved
**jointly** across groups. Either (i) anchor ≥ `D+1` common items to a fixed reference configuration and
Procrustes-map every group to it, or (ii) estimate all groups in one shared space with common item
parameters (HLSIRM route). Anchoring is what makes group centroids comparable.

---

## 6. Multidimensional extension (MIRT) and its relation to LSIRM

### 6.1 Compensatory MIRT (Reckase, 2009) `[S]`

For a `d`-dimensional trait `θ_p∈ℝ^d`, item slope vector `a_i∈ℝ^d`, intercept `d_i`:
$$
P(Y_{pi}=1\mid\theta_p)=c_i+(1-c_i)\,\operatorname{logit}^{-1}\!\big(a_i^\top\theta_p+d_i\big).
$$
Summary indices:
$$
\text{MDISC}_i=\lVert a_i\rVert=\sqrt{\textstyle\sum_{m=1}^d a_{im}^2},\qquad
\text{MDIFF}_i=\frac{-d_i}{\lVert a_i\rVert},\qquad
\text{direction cosines }=\frac{a_{im}}{\text{MDISC}_i}.
$$
"Compensatory" because a low coordinate of `θ_p` can be offset by a high one through the inner product.
Estimated by MML/EM or MH-RM; identifiability fixed by rotation constraints (as in factor analysis).

### 6.2 How LSIRM relates `[~]`

- The **multiplicative** LSIRM term `z_p^⊤ w_i` is algebraically a MIRT compensatory term with item
  "loadings" `= w_i` and person "traits" `= z_p` (a bilinear/eigenmodel factorization, as HLSIRM uses).
  So a `D`-dimensional inner-product LSIRM ≈ a `D`-dimensional compensatory MIRT with the main effects
  `θ_p,β_i` as an extra rank-one term.
- The **distance** LSIRM term `−γ\lVert z_p-w_i\rVert` is **non-compensatory / ideal-point-like**: the
  probability is maximized when the respondent sits *at* the item's location and falls off symmetrically in
  every direction — closer to an unfolding model than to a monotone MIRT surface. Jeon et al.'s
  "Multidimensional Latent Space Item Response Models: A Note on the Relativity of Conditional Dependence"
  discusses how the recovered map dimension and conditional-dependence structure are only defined **relative**
  to a reference, reinforcing that `D` is a modeling choice validated by fit, not an absolute count.
- **Choosing `D`:** fit `D=1,2,3,…` and compare by BIC / WAIC / cross-validated log-likelihood (as the
  package does with BIC); interpretability usually caps at `D=2`.

---

## 7. Item-fit statistics

### 7.1 Orlando & Thissen (2000) S-X² with the Lord–Wingersky recursion `[V formula/df, S recursion]`

Group examinees by **observed summed score** `s∈{1,…,I-1}` (score-independent of `θ̂`). For item `i`:
$$
\boxed{\;S\text{-}X^2_i=\sum_{s=1}^{I-1} N_s\,\frac{\big(O_{is}-E_{is}\big)^2}{E_{is}\,(1-E_{is})}\;},
\qquad df = (I-1)-m_i,
$$
where `N_s` = number of examinees with summed score `s`, `O_{is}` = observed proportion correct on item `i`
in score group `s`, `E_{is}` = model-expected proportion, and `m_i` = number of estimated parameters for
item `i` (1 for Rasch, 2 for 2PL, 3 for 3PL). Score groups `0` and `I` are excluded (trivial proportions).

**Expected proportion `E_{is}`** (this is where the recursion enters):
$$
E_{is}=\frac{\displaystyle\int P_i(\theta)\,S_{s-1}^{(-i)}(\theta)\,g(\theta)\,d\theta}
{\displaystyle\int S_{s}(\theta)\,g(\theta)\,d\theta},
$$
i.e. `E_{is}=P(\text{item }i\text{ correct}\mid \text{summed score}=s)`: numerator = P(item `i` correct **and**
total `= s`) — if item `i` is correct the *other* `I-1` items must sum to `s-1`; denominator = P(total `= s`).
Both integrals are evaluated by Gauss–Hermite quadrature over `g(θ)`.

**Lord–Wingersky (1984) recursion** for the summed-score likelihood at fixed `θ`. Let
`f_r^{(n)}(θ)=P(\text{score}=r\text{ using items }1..n\mid θ)`:
$$
f_0^{(1)}=1-P_1(\theta),\quad f_1^{(1)}=P_1(\theta);\qquad
f_r^{(n)}(\theta)=f_r^{(n-1)}(\theta)\big(1-P_n(\theta)\big)+f_{r-1}^{(n-1)}(\theta)\,P_n(\theta),
$$
for `n=2,…,I` and `r=0,…,n` (with `f_r^{(n-1)}≡0` for `r<0` or `r>n-1`). Then
`S_s(θ):=f_s^{(I)}(θ)`, and `S_{s-1}^{(-i)}(θ)` is the same recursion run over the `I-1` items **excluding
item `i`**. (Compute the leave-one-out distributions by removing each item's factor.)

**Generalization.** Kang & Chen (2008) extend S-X² to polytomous / graded response models (bins on the
total summed score, cell probabilities via a generalized Lord–Wingersky recursion). The likelihood-ratio
analogue is `S-G²_i = 2 Σ_s N_s[ O_{is} ln(O_{is}/E_{is}) + (1-O_{is}) ln((1-O_{is})/(1-E_{is})) ]`.

### 7.2 Infit / Outfit mean squares (Wright & Masters, 1982) `[S/V heuristics]`

Standardized residual for the (p,i) cell (`E_{pi}=P_{pi}`, variance `W_{pi}=P_{pi}(1-P_{pi})` for
dichotomous; `W_{pi}=Σ_k (k-E_{pi})^2 P_{pik}` for polytomous):
$$
z_{pi}=\frac{y_{pi}-E_{pi}}{\sqrt{W_{pi}}}.
$$
Per-**item** fit:
$$
\text{Outfit}_i=\frac{1}{N}\sum_{p=1}^{N} z_{pi}^2
=\frac1N\sum_p \frac{(y_{pi}-E_{pi})^2}{W_{pi}},\qquad
\text{Infit}_i=\frac{\sum_{p}(y_{pi}-E_{pi})^2}{\sum_{p} W_{pi}}
=\frac{\sum_p W_{pi}z_{pi}^2}{\sum_p W_{pi}}.
$$
Per-**person** fit uses the same expressions summing over items `i` at fixed `p`. Outfit is the unweighted
mean square (sensitive to outliers on items far from a person's ability); Infit is
**information-weighted** (down-weights those extremes). Expected value ≈ 1. Optional
**Wilson–Hilferty** standardization to an approximately `N(0,1)` `t`:
$$
t=\Big(\text{MS}^{1/3}-1\Big)\frac{3}{q}+\frac{q}{3},\qquad q^2=\widehat{\operatorname{Var}}(\text{MS}).
$$

### 7.3 Posterior predictive checks (Bayesian item fit) `[S]`

Draw `y^{rep}` from the posterior predictive; discrepancy `T(y,ζ)` (e.g. item odds-ratios, item-total
correlations, χ² by score group); posterior predictive `p`-value
`ppp = P\big(T(y^{rep},ζ)\ge T(y,ζ)\mid y\big)`, estimated as the fraction of MCMC draws with
`T(y^{rep(s)},ζ^{(s)})≥T(y,ζ^{(s)})`. Values near 0 or 1 flag misfit (Sinharay, 2005). The `lsirm12pl`
`gof()` (observed vs. replicated item means; ROC/AUC) is a lightweight instance.

---

## 8. Person-fit statistics

### 8.1 `l_z` (Drasgow, Levine & Williams, 1985) `[V]`

Standardized log-likelihood of a response pattern at (estimated) ability `θ`:
$$
l(\theta)=\sum_{i=1}^{n}\Big\{X_i\log\frac{P_i(\theta)}{1-P_i(\theta)}+\log\big(1-P_i(\theta)\big)\Big\},
$$
$$
l_z(\theta)=\frac{l(\theta)-E[l(\theta)]}{\sqrt{\operatorname{Var}[l(\theta)]}}
=\frac{\sum_{i}(X_i-P_i(\theta))\log\frac{P_i(\theta)}{1-P_i(\theta)}}{\sqrt{\operatorname{Var}[l(\theta)]}},
$$
with `E[l(θ)]=Σ_i[P_i\log P_i+(1-P_i)\log(1-P_i)]` and
`Var[l(θ)]=Σ_i P_i(1-P_i)\big(\log\frac{P_i}{1-P_i}\big)^2`. Under the model with **known** `θ`, `l_z ≈ N(0,1)`;
low (very negative) values flag aberrance. **Problem:** substituting `θ̂` biases the mean/variance so the
`N(0,1)` reference is wrong — corrected by `l_z^*`.

### 8.2 Snijders (2001) `l_z^*` — asymptotically correct standardization `[V]`

Snijders' general class of person-fit statistics:
`W(\theta)=Σ_{i=1}^{n}(X_i-P_i(\theta))\,w_i(\theta)` (for `l_z`, `w_i(θ)=\log\frac{P_i(θ)}{1-P_i(θ)}`).
Define, with `P_i'(θ)=dP_i/dθ` and Fisher information `I(θ)=Σ_i \frac{P_i'(θ)^2}{P_i(θ)(1-P_i(θ))}`:
$$
r_i(\theta)=\frac{P_i'(\theta)}{P_i(\theta)\{1-P_i(\theta)\}},\qquad
c(\theta)=\frac{\sum_i P_i'(\theta)\,w_i(\theta)}{\sum_i P_i'(\theta)\,r_i(\theta)}
=\frac{1}{I(\theta)}\sum_i P_i'(\theta)\,w_i(\theta),
$$
$$
\tilde w_i(\theta)=w_i(\theta)-c(\theta)\,r_i(\theta),\qquad
\tau^2(\theta)=\frac1n\sum_i \tilde w_i^2(\theta)\,P_i(\theta)\{1-P_i(\theta)\}.
$$
The corrected statistic (asymptotically `N(0,1)` even with estimated `θ̂`):
$$
\boxed{\;l_z^*=\tilde Z(\hat\theta)=\frac{W(\hat\theta)+c(\hat\theta)\,r_0(\hat\theta)}{\sqrt{n}\,\tau(\hat\theta)}\;}
$$
where the estimator-dependent term `r_0(θ̂)` is:
$$
r_0(\hat\theta)=\begin{cases}
0, & \text{MLE},\\[2pt]
\dfrac{d\log f(\hat\theta)}{d\hat\theta}, & \text{MAP (prior } f),\\[6pt]
\dfrac{J(\hat\theta)}{2\,I(\hat\theta)}, & \text{WLE (Warm), with } J(\theta)=\sum_i \dfrac{P_i'P_i''}{P_i(1-P_i)}.
\end{cases}
$$
Substituting `w_i=\log\frac{P_i}{1-P_i}` gives the corrected `l_z^*`. **Scope:** derived for dichotomous
items; the mean/variance correction absorbs the first-order effect of estimating `θ`. (Multidimensional /
polytomous / mixed-type extensions: Sinharay, 2016; and the "corrected version" note, arXiv:2605.00216,
which is the source of the formulas above.)

---

## 9. Item selection / removal decision procedure (grounded in cited literature)

A defensible LSIRM item-screening pipeline, combining classical fit rules with LSIRM-specific map
diagnostics. Apply after a converged fit; re-estimate after each removal round (fit indices shift).

1. **S-X² misfit, multiplicity-controlled.** Compute `S-X²_i` (§7.1) and its `p`-value for every item.
   Control the false discovery rate across items with **Benjamini–Hochberg** (1995): sort `p_{(1)}≤…≤p_{(I)}`,
   reject where `p_{(i)}≤ (i/I)·q` (`q=.05`). Flag rejected items. (S-X² is preferred over `θ̂`-binned Q₁/G²
   because its summed-score bins are model-independent; Orlando & Thissen, 2000.)
2. **Infit/Outfit out of range.** Flag items with mean squares outside a productive-misfit band.
   Wright & Linacre (1994) "reasonable" ranges: high-stakes MCQ ≈ `[0.8, 1.2]`; a common working band is
   **`[0.7, 1.3]`**; lenient `[0.5, 1.5]` (de Ayala, 2009). Values `>` upper bound (underfit) are the
   serious ones — the item is noisier than the model expects (degrades measurement); values `<` lower bound
   (overfit) are redundant but rarely harmful. Optionally use the standardized `t` with `|t|>2`, but note
   `t` is over-powered at large `N` (Bond & Fox, 2007) — prefer the mean-square band there.
3. **Low discrimination (2PL/MIRT).** Flag items with `α_i` (or `\text{MDISC}_i=\lVert a_i\rVert`) below a
   threshold (e.g. `< 0.3–0.4` on the logistic metric); such items carry little information and often
   coincide with S-X² misfit.
4. **Person-fit screen before item decisions.** Remove or down-weight aberrant respondents flagged by
   `l_z^* < -1.645` (one-sided 5%) *before* finalizing item removals, so item statistics are not distorted
   by cheating/careless patterns (Snijders, 2001; §8.2).
5. **LSIRM-specific map diagnostics.**
   - **Isolated items.** An item whose position `w_i` is far from the bulk of respondent positions `{z_p}`
     (large `γ`-weighted distance `γ·\text{mean}_p \lVert z_p-w_i\rVert`, i.e. a large interaction penalty for
     nearly everyone) discriminates poorly in the region where data live — a candidate for removal or
     rewording. This is the LSIRM reading of an item that "no one interacts with."
   - **Interaction necessity.** If, after refit, an item's removal barely changes `γ` and the map
     (or a spike-and-slab prior keeps `γ≈0` for that item), the interaction term is not needed — the item is
     adequately described by `θ_p+β_i` alone.
   - **Interaction DIF (multigroup).** Flag items whose group-specific positions differ,
     `\lVert w_i^{(g)}-w_i^{(g')}\rVert` large (§5.2).
6. **Decision.** Remove an item only when it fails **multiple** criteria (e.g. BH-significant S-X²
   **and** Infit/Outfit out of band, or low `α_i` **and** map-isolated), and when removal is substantively
   defensible (content coverage preserved). Document each removal and re-run item- and person-fit on the
   reduced set. Prefer revision over deletion when content is essential.

---

## 10. Minimal implementation checklist

- **Likelihood kernel:** `logit⁻¹(θ_p+β_i−γ‖z_p−w_i‖)` (binary) or Gaussian mean (continuous); cache
  distances `d_{pi}` and unit vectors `u_{pi}`.
- **Estimator:** MH-RM (§3.C) as the default MML engine (handles the `(1+D)`-dim per-person + `D`-dim
  per-item latents without quadrature blow-up); JML (§3.B) for a warm start; Bock–Aitkin quadrature
  (§3.A.1) only for the `D`-free trait margin / plain 2PL calibration.
- **Identifiability each iteration:** center `Z, W`; Procrustes-rotate `W` to a fixed reference `W₀`;
  fix `γ` scale or standardize position variance. Anchor `≥ D+1` items for multigroup comparability.
- **Variance components:** update `σ², σ_ε²` (and multilevel `σ_α², τ², Ψ_z`) by random-effects EM moment
  equations / conjugate draws.
- **Fit module:** S-X² via Lord–Wingersky recursion (§7.1); Infit/Outfit (§7.2); `l_z^*` (§8.2);
  posterior-predictive checks if a Bayesian variant is also run.
- **Screening:** the §9 pipeline with BH-FDR on S-X² and a `[0.7,1.3]` mean-square band.

---

## 11. Model-equation quick index

| Model | Core equation |
|---|---|
| LSIRM (distance) | `logit P = θ_p + β_i − γ‖z_p−w_i‖` |
| LSIRM (multiplicative) | `logit P = θ_p + β_i + z_p^⊤w_i` |
| 2PL LSIRM | `logit P = α_i θ_p + β_i − γ‖z_p−w_i‖`, `α_1=1` |
| Continuous LSIRM | `Y_{pi} = α_i θ_p + β_i − γ‖z_p−w_i‖ + ε_{pi}`, `ε∼N(0,σ_ε²)` |
| HLSIRM (multilevel) | `logit P = α_{i(k)} + β_j + z_{i(k)}^⊤w_j + ε`, `α_{i(k)}∼N(α_{(k)},σ²_{(k)})` |
| Multigroup IRT | `θ_{pg}∼N(μ_g,σ_g²)`, anchored items, `μ_1=0,σ_1²=1` |
| MIRT (compensatory) | `logit P = a_i^⊤θ_p + d_i`, `MDISC=‖a_i‖`, `MDIFF=−d_i/‖a_i‖` |
| MMLE (persons out) | `L=∏_p ∫∫ ∏_i P_{pi}^{y}(1−P_{pi})^{1−y} φ(θ_p)φ_D(z_p)dz_p dθ_p` |
| MH-RM update | `ξ^{(t)}=ξ^{(t−1)}+ε_t H⁻¹ s^{(t)}`, `s=∇_ξ log f(y,φ^{(t)};ξ)` |
| S-X² | `Σ_s N_s (O_{is}−E_{is})²/[E_{is}(1−E_{is})]`, `df=(I−1)−m_i` |
| Lord–Wingersky | `f_r^{(n)}=f_r^{(n−1)}(1−P_n)+f_{r−1}^{(n−1)}P_n` |
| Outfit / Infit | `N⁻¹Σ_p z_{pi}²` / `Σ_p(y−E)²/Σ_p W` |
| `l_z` | `(l−E[l])/√Var[l]` |
| `l_z^*` | `[W(θ̂)+c(θ̂)r_0(θ̂)]/[√n·τ(θ̂)]` |

---

## 12. Citations

**Verified verbatim online in this compilation `[V]`:**

- Jeon, M., Jin, I.-H., Schweinberger, M., & Baugh, S. (2021). *Mapping Unobserved Item–Respondent
  Interactions: A Latent Space Item Response Model with Interaction Map.* **Psychometrika, 86**(2), 378–403.
  DOI: 10.1007/s11336-021-09762-5. arXiv:2007.08719. — model, priors, MCMC, Procrustes verified.
- Go, D., Kim, G., Park, J., Park, J., Jeon, M., & Jin, I. H. (2025). *lsirm12pl: An R package for latent
  space item response modeling.* **The R Journal** (contributed). arXiv:2205.06989. — 2PL/continuous
  variants, priors, MH-within-Gibbs, BIC verified. Code: https://github.com/jiniuslab/lsirm12pl
- Park, J., Shin, ..., Jeon, M., Kim, ..., & Jin, I. H. (2026). *Hierarchical Latent Space Item Response
  Model for Analyzing Mental Health Vulnerability of Elementary School Students in South Korea.*
  arXiv:2603.13677 (DOI 10.48550/arXiv.2603.13677). — full multilevel equations verified.
- Snijders, T. A. B. (2001). *Asymptotic null distribution of person fit statistics with estimated person
  parameter.* **Psychometrika, 66**(3), 331–342. DOI: 10.1007/BF02294437. — `l_z^*` correction formulas
  verified via the corrected re-derivation, arXiv:2605.00216 ("Simplicity Above Elegance…", 2026).
- Orlando, M., & Thissen, D. (2000). *Likelihood-Based Item-Fit Indices for Dichotomous Item Response
  Theory Models.* **Applied Psychological Measurement, 24**(1), 50–64. DOI: 10.1177/01466216000241003. —
  S-X² formula and `df` verified (NCME Module 40; CRAN `CDM::itemfit.sx2`).
- Cai, L. (2010). *High-Dimensional Exploratory Item Factor Analysis by a Metropolis–Hastings Robbins–Monro
  Algorithm.* **Psychometrika, 75**(1), 33–57. DOI: 10.1007/s11336-009-9136-x. (Companion: *MH-RM for
  Confirmatory Item Factor Analysis*, **J. Educ. Behav. Stat., 35**(3), 307–335,
  DOI: 10.3102/1076998609353115.) — algorithm description verified.

**Standard results reproduced from memory (source cited, exact page not re-fetched) `[S]`:**

- Bock, R. D., & Aitkin, M. (1981). *Marginal maximum likelihood estimation of item parameters:
  Application of an EM algorithm.* **Psychometrika, 46**(4), 443–459. DOI: 10.1007/BF02293801. (Errata
  47, 369.) — EM/quadrature `E`-step counts `N̄_q, r̄_{iq}` and `M`-step equations are standard;
  the primary PDF could not be text-extracted cleanly online (existence/description verified only).
- Lord, F. M., & Wingersky, M. S. (1984). *Comparison of IRT true-score and equipercentile observed-score
  equatings.* **Applied Psychological Measurement, 8**(4), 453–461. DOI: 10.1177/014662168400800409. —
  the summed-score recursion is referenced by Orlando & Thissen; exact recursion from memory.
- Wright, B. D., & Masters, G. N. (1982). *Rating Scale Analysis.* MESA Press. — infit/outfit mean squares.
- Wright, B. D., & Linacre, J. M. (1994). *Reasonable mean-square fit values.* **Rasch Measurement
  Transactions, 8**(3), 370. — the `[0.5,1.5]`/`[0.7,1.3]`/`[0.8,1.2]` bands used in §9.
- Wright, B. D., & Panchapakesan, N. (1969). *A procedure for sample-free item analysis.* **Educational
  and Psychological Measurement, 29**, 23–48. — origin of infit/outfit (per NCME Module 40 `[V]`).
- Drasgow, F., Levine, M. V., & Williams, E. A. (1985). *Appropriateness measurement with polychotomous
  item response models and standardized indices.* **British J. Math. Stat. Psychology, 38**, 67–86.
  DOI: 10.1111/j.2044-8317.1985.tb00817.x. — `l_z` (base `l_z` formula also `[V]` via the Snijders source).
- Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). *Latent space approaches to social network
  analysis.* **JASA, 97**(460), 1090–1098. DOI: 10.1198/016214502388618906. — latent-space distance model,
  JML/MLE-of-positions, and the translation/rotation/reflection identifiability that LSIRM inherits.
- Fox, J.-P., & Glas, C. A. W. (2001). *Bayesian estimation of a multilevel IRT model using Gibbs
  sampling.* **Psychometrika, 66**(2), 271–288. DOI: 10.1007/BF02294839. Fox, J.-P. (2010). *Bayesian Item
  Response Modeling.* Springer. DOI: 10.1007/978-1-4419-0742-4. — two-level IRT structural equations.
- Bock, R. D., & Zimowski, M. F. (1997). *Multiple group IRT.* In van der Linden & Hambleton (Eds.),
  *Handbook of Modern Item Response Theory* (pp. 433–448). Springer. DOI: 10.1007/978-1-4757-2691-6_25. —
  multigroup means/variances, anchoring, DIF.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory.* Springer.
  DOI: 10.1007/978-0-387-89976-3. — compensatory MIRT, MDISC/MDIFF.
- Gower, J. C. (1975). *Generalized Procrustes analysis.* **Psychometrika, 40**(1), 33–51.
  DOI: 10.1007/BF02291478. — identifiability resolution.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate.* **JRSS-B, 57**(1), 289–300.
  DOI: 10.1111/j.2517-6161.1995.tb02031.x. — multiplicity control in §9.
- de Ayala, R. J. (2009). *The Theory and Practice of Item Response Theory.* Guilford. — fit-flag heuristics.
- Sinharay, S. (2005). *Assessing fit of unidimensional item response theory models using a Bayesian
  approach.* **J. Educ. Measurement, 42**(4), 375–394. DOI: 10.1111/j.1745-3984.2005.00021.x. — PPMC.

**Located but details not fetched (cite for the relevant sub-topic) `[~]`:**

- Jeon, M., et al. *Multidimensional Latent Space Item Response Models: A Note on the Relativity of
  Conditional Dependence.* **Psychometrika** (Cambridge Core). — §6.2 relation of `D` to MIRT.
- Kang, I., & Jeon, M. (2024). Interaction-map summary quantities (`α̃, β̃`), cited by HLSIRM §4.1.
- Salter-Townshend, M., & Murphy, T. B. (2013). *Variational Bayesian inference for the latent position
  cluster model.* **Computational Statistics & Data Analysis, 57**, 661–671.
  DOI: 10.1016/j.csda.2012.08.004. — VI for latent-space models (§3.D).
- Kang, T., & Chen, T. T. (2008). *Performance of the generalized S-X² for polytomous IRT.*
  **J. Educ. Measurement, 45**, 391–406. — polytomous S-X² (§7.1).

---

### Verification summary
Directly verified online: LSIRM base model/priors/MCMC/Procrustes (arXiv:2007.08719); 2PL & continuous
LSIRM + priors + estimation (arXiv:2205.06989); full multilevel HLSIRM (arXiv:2603.13677); Snijders `l_z^*`
exact formulas (arXiv:2605.00216); S-X² formula + df and infit/outfit (NCME Module 40 + CRAN CDM);
MH-RM description (Cai 2010, Springer/SAGE). Reproduced from standard sources (existence confirmed, exact
symbols from memory): Bock–Aitkin EM update equations, Lord–Wingersky recursion, infit/outfit MSQ formulas,
Fox–Glas multilevel IRT, Bock–Zimowski multigroup, Reckase MIRT indices, `l_z` base. A dedicated
**multigroup-LSIRM** paper was **not** found online — §5.2 is a construction by analogy (HLSIRM + Bock–
Zimowski), flagged as such.
