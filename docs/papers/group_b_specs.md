# Group B: Implementation Specs for `fast-mlsirm`

**Target engine.** `fast-mlsirm` — Rust marginal-EM (Bock–Aitkin MMLE) engine for latent-space IRT.
Working model for a binary response `Y_pi ∈ {0,1}` of person `p` to item `i`:

$$
\operatorname{logit}\big(P(Y_{pi}=1)\big)=a_i\,\theta_{p,d(i)}+b_i-\gamma\,\lVert \xi_p-\zeta_i\rVert ,
$$

with discrimination `a_i`, easiness `b_i`, simple-structure loading map `d(i)` (item `i` → one latent
dimension), latent respondent/item positions `ξ_p, ζ_i ∈ ℝ^D`, distance weight `γ ≥ 0`. Population:
`θ_{p,d} ~ N(μ_{g(p),d}, σ²_{g(p),d})` (multigroup means/SDs), optional multilevel random intercept
`u_c ~ N(0, σ²_u)` added to the linear predictor. E-step: tensor Gauss–Hermite (`D ≤ 3`) or Halton-QMC /
MC-EM over the trait, Bock–Aitkin expected counts ("artificial data"). M-step: per-item Newton on the
expected binomial log-likelihood; `γ` by 1-D Newton; `μ_{gd}, σ²_{gd}` from posterior moments; `σ²_u`
likewise. Missingness handled "not-presented" (missing `(p,i)` cells dropped from the E-step). Item
anchoring / FIPC: `PopulationSpec::SingleFree` + common (anchored) item block, reference group `N(0,1)`.

**Notation bridge.** Each paper's original symbols are kept in its equations; the "map to engine" text
translates to `(θ, a_i, b_i, γ, ξ, ζ, μ_{gd}, σ_{gd})`. Sign convention: the engine uses **easiness**
`b_i` (`+b_i` in the linear predictor); papers that use **difficulty** `β_i` map by `b_i = −β_i`.

**Feasibility legend.** `direct` = expressible with existing knobs or a thin preprocessing layer;
`adaptation` = new E-/M-step term or moment update; `already-covered` = engine already does the core;
`document-only` = record the mapping, no code.

---

## 1. Perumean-Chaney, Morgan, McDowall & Aban (2013) — zero inflation / overdispersion

**Citation.** Perumean-Chaney, S. E., Morgan, C., McDowall, D., & Aban, I. (2013). Zero-inflated and
overdispersed: what's one to do? *Journal of Statistical Computation and Simulation*, 83(9), 1671–1683.
https://doi.org/10.1080/00949655.2012.668550

### 1.1 The paper's models (count data), exact

Poisson (their Eq. 1):
$$
\Pr(Y=y)=\frac{e^{-\mu}\mu^{y}}{y!},\qquad y=0,1,2,\dots,\qquad E(Y)=\operatorname{Var}(Y)=\mu .
$$

Negative binomial (Eq. 2), mean `μ`, `Var(Y)=μ+μ²/θ`:
$$
\Pr(Y=y)=\frac{\Gamma(\theta+y)}{\Gamma(\theta)\,\Gamma(y+1)}\,
\frac{\theta^{\theta}\,\mu^{y}}{(\theta+\mu)^{\theta+y}},\qquad y=0,1,2,\dots
$$

Two-component mixture with mixing proportion `p` (Eq. 3), the general zero-inflation device:
$$
P(Y=y)=p\cdot g_1(y)+(1-p)\cdot g_2(y).
$$

Zero-inflated Poisson (ZIP, Eq. 4), `g_1` degenerate at 0, mixing proportion `π`:
$$
P(Y=0)=\pi+(1-\pi)\,e^{-\mu},\qquad
P(Y=y)=(1-\pi)\,\frac{\mu^{y}e^{-\mu}}{y!},\quad y=1,2,\dots
$$

Zero-inflated negative binomial (ZINB, Eq. 5):
$$
P(Y=0)=\pi+(1-\pi)\,\frac{\theta^{\theta}}{(\theta+\mu)^{\theta}},\qquad
P(Y=y)=(1-\pi)\,\frac{\Gamma(\theta+y)}{\Gamma(\theta)\,\Gamma(y+1)}\,
\frac{\theta^{\theta}\mu^{y}}{(\theta+\mu)^{\theta+y}},\quad y=1,2,\dots
$$

The zero-inflation `π` is the probability of belonging to a **structural-zero** ("never at risk") class;
the `(1−π)` class is "at risk" and may or may not produce a zero.

**Practical recommendations (verbatim thrust).** (i) Ignoring zero inflation (fitting Poisson/NB to a
ZI process) **underestimates the mean → misses significant findings (Type II)**. (ii) Ignoring
overdispersion *within* the ZI data (fitting ZIP when the truth is ZINB) **overestimates the mean and
shrinks the SE → false positives (Type I)**. (iii) **When unsure whether ZIP or ZINB, use ZINB** (wider
CIs, robust to unmodeled overdispersion). (iv) Small mean/`N ≤ 50` destabilizes ZINB; prefer `N > 50`.
(v) The two-step LRT–Vuong selector is unreliable at small mean/`N` (correctly IDs ZIP only at moderate
mean μ=5 and N=100, poor for ZINB); mixture LRT asymptotics are not standard, so a naive chi-square test
of `π=0` is untrustworthy.

### 1.2 Mapping to a zero-inflated IRT mixture

The count "mean `μ`" has no IRT analog; the transferable structure is the **membership mixture on the
all-zero response pattern**. Let `y_p = (y_{p1},…,y_{pI})` and `A_p = 1[y_p = 0]` (all items zero). Design:

$$
\boxed{\;
L_p=\pi\cdot \mathbf 1[y_p=\mathbf 0]+(1-\pi)\,L_{\mathrm{IRT}}(y_p),
\qquad
L_{\mathrm{IRT}}(y_p)=\int \prod_{i\in\Omega_p} P_i(\theta)^{y_{pi}}\big(1-P_i(\theta)\big)^{1-y_{pi}}\,g(\theta)\,d\theta
\;}
$$

with `P_i(θ) = logit^{-1}(a_i θ_{d(i)} + b_i − γ‖ξ_p−ζ_i‖)` the engine kernel and `Ω_p` the observed
cells. This is the **exact IRT counterpart of Eq. 4/5**: the degenerate `g_1` (spike at "all-zero") plays
the role of the count spike at `y=0`; the "at-risk" class is the LSIRM. Only respondents with `A_p=1` get
probability mass from the spike — anyone with a single `y_{pi}=1` has spike probability 0.

**How the findings map to the design.**

| Paper finding (count) | IRT-mixture consequence |
|---|---|
| Ignoring `π` underestimates `μ`, causes Type II | Forcing `π=0` (plain LSIRM) inflates estimated **easiness `b_i`** (all-zero people look like low-trait people, dragging item easiness / trait mean down); genuine effects can be masked. |
| ZIP-when-ZINB overestimates `μ`, Type I | The IRT analog of "extra within-class dispersion" is an **over-thin at-risk population model**. Forcing a single tight `N(μ_g,σ_g²)` when the at-risk class needs heavier tails / a random intercept is the "ZIP-when-ZINB" error → over-confident item SEs. **Default to the richer at-risk model** (keep multilevel `σ_u`, do not fix `σ_g`), the direct analog of "prefer ZINB." |
| LRT for `π=0` unreliable (mixture boundary) | Test `π=0` with a **boundary-corrected** statistic (50:50 mixture of `χ²_0` and `χ²_1`) or a parametric bootstrap, never a naive `χ²_1`. |
| Small mean/N destabilizes | `π` is identified **only from all-zero patterns**; short tests, high-difficulty items, or few all-zero respondents give weak `π` identification — the exact analog of their small-`μ` instability. Warn/shrink when the all-zero count is small. |

**`π` estimation inside the EM (structural-zero class).** Add a class indicator `C_p ∈ {S,R}` (structural
zero / at-risk) as a second latent layer above the trait.

- **E-step (class responsibility).** For every respondent,
  $$
  r_p \;=\; P(C_p=S\mid y_p)\;=\;
  \frac{\pi\,\mathbf 1[y_p=\mathbf 0]}{\pi\,\mathbf 1[y_p=\mathbf 0]+(1-\pi)\,L_{\mathrm{IRT}}(y_p)} ,
  $$
  so `r_p = 0` whenever `A_p = 0` (any observed 1). `L_IRT(y_p)` for an all-zero respondent is just the
  already-computed marginal `∑_v w_v ∏_i(1−P_i(v))` over quadrature nodes `v`.
- **E-step (trait posterior), reweighted.** Person `p` contributes its Bock–Aitkin expected counts to the
  item tables with weight `(1−r_p)` (at-risk responsibility). Non-all-zero respondents are unchanged
  (`1−r_p=1`).
- **M-step.** `π ← (1/N) Σ_p r_p`. Item/`γ`/population updates are the existing M-step run on the
  `(1−r_p)`-weighted expected counts. Optional covariate model `π_p = logit^{-1}(w_p'η)` replaces the
  scalar update with a 1-step IRLS on `η` against targets `r_p` (mirrors Lambert's logit zero model).

### 1.3 Implementation plan

- **E-step change:** one extra scalar per respondent (`r_p`); reuses the existing all-zero marginal. No new
  integration.
- **M-step change:** one closed-form update for `π` (or a small IRLS for `η`); multiply existing expected
  counts by `(1−r_p)`.
- **Parameter count:** `+1` (`π`) or `+q` (covariate zero model). Everything else unchanged.
- **Identification:** `π=0` nests the plain LSIRM. Needs a non-trivial number of all-zero patterns; if the
  observed all-zero count is 0, `π` is unidentified → clamp to 0 and warn. Keep the reference group `N(0,1)`.
- **Minimal recovery test:** simulate `N=2000`, `I=20`, spike fraction `π∈{0.2,0.4}`, at-risk from the
  LSIRM; fit (a) plain LSIRM and (b) the mixture. Assert: `π̂` within ±0.03 of truth; mixture recovers
  `b_i` while plain LSIRM shows systematic easiness bias of the sign predicted above; boundary-corrected
  LRT rejects `π=0` when `π=0.4` and holds nominal size when `π=0`.

### 1.4 Out of scope

Count/ordinal responses (Poisson/NB IRT), the `μ`/`θ`-overdispersion parameter itself, hurdle models
(which differ from zero-inflation: hurdle has no "at-risk zeros"), and the LRT–Vuong selection study —
we adopt the paper's *conclusion* (prefer the richer model, boundary-correct the test), not its selector.

---

## 2. Jeon, Rijmen & Rabe-Hesketh (2013) — multiple-group bifactor DIF

**Citation.** Jeon, M., Rijmen, F., & Rabe-Hesketh, S. (2013). Modeling Differential Item Functioning
Using a Generalization of the Multiple-Group Bifactor Model. *Journal of Educational and Behavioral
Statistics*, 38(1), 32–60. https://doi.org/10.3102/1076998611432173

### 2.1 The paper's models, exact

Multiple-group unidimensional 2PL DIF (Eq. 1), person `j` in group `h`, item `i`:
$$
\operatorname{logit}\big(\Pr(y_{j(h)i}=1\mid \theta_{j(h)})\big)=a_i\big(\theta_{j(h)}s_h+\mu_h\big)+b_i+d_{ih},
$$
with `θ_{j(h)} ~ N(0,1)`; the realized ability is `θ*_{j(h)} = θ_{j(h)}s_h + μ_h` (group mean `μ_h`, SD
`s_h`). `d_{ih}` is **uniform DIF** (a group-`h` shift in item easiness). Reference group `h=1`:
`μ_1=0, s_1=1, d_{i1}=0`. Anchor (DIF-free) items: `d_{ih}=0 ∀h`.

Multiple-group **bifactor** DIF (Eq. 2), item `i` in testlet `k`, general `g` and specific `k` dims:
$$
\operatorname{logit}\big(\Pr(y_{j(h)i(k)}=1\mid \theta^{*}_{j(h)g},\theta^{*}_{j(h)k})\big)
=a_{ig}\,\theta^{*}_{j(h)g}+a_{ik}\,\theta^{*}_{j(h)k}+b_i+d_{ih}.
$$

Independent-dimension parameterization (Eq. 3): `θ*_{j(h)g}=θ_{j(h)g}s_{gh}+μ_{gh}`,
`θ*_{j(h)k}=θ_{j(h)k}s_{kh}+μ_{kh}`.

Correlation (Cholesky) parameterization (Eq. 4), relaxing orthogonality to *conditional* independence of
the specific dims given the general dim:
$$
\theta^{*}_{j(h)g}=\theta_{j(h)g}\,c_{ggh}+\mu_{gh},\qquad
\theta^{*}_{j(h)k}=\theta_{j(h)g}\,c_{gkh}+\theta_{j(h)k}\,c_{kkh}+\mu_{kh},
$$
with `C_h` lower-triangular (nonzero off-diagonals only in the first column) and `Σ_h = C_h C_h'`.

**Identification.** Reference group: all means 0, all variances 1, all covariances 0
(`μ_{g1}=μ_{k1}=0, c_{gg1}=c_{kk1}=1, c_{km1}=0`). Anchor set must contain **≥1 item per testlet**.
**DIF testing:** `H_0: d_{ih}=0` by **Wald or likelihood-ratio test** (asymptotically equivalent; the paper
notes LR/Wald discrepancy grows when the log-likelihood is non-quadratic). Item **purification** is
recommended so the anchor set is itself DIF-free.

### 2.2 Mapping to the engine

The engine is **simple-structure**, not bifactor, so the general+specific correlated-dimension machinery
(Eq. 2–6) and its junction-tree E-step are **not** the port. The transferable, engine-shaped piece is the
**multiple-group DIF layer** (Eq. 1) on top of the existing multigroup means/SDs and anchoring:

$$
\operatorname{logit}\big(P(Y_{pi}=1)\big)=a_i^{g(p)}\,\theta_{p,d(i)}+b_i^{g(p)}-\gamma\lVert\xi_p-\zeta_i\rVert,
\qquad \theta_{p,d}\sim N(\mu_{g,d},\sigma_{g,d}^2).
$$

- **Anchor items** carry group-common `(a_i, b_i)` (the existing FIPC/anchor block).
- **Studied (candidate-DIF) items** get a **group-specific easiness `b_i^g`** (uniform DIF `d_{ih}`), and
  optionally a **group-specific slope `a_i^g`** (non-uniform DIF). In the engine, "studied item = non-anchor
  with per-group parameters"; "anchor = common parameter." This is already close to how anchoring vs.
  free items are represented — DIF adds the *per-group* free parameter on flagged items only.
- Impact is absorbed by the existing `μ_{g,d}, σ_{g,d}` — the paper's central point that impact must be
  modeled to avoid spurious DIF is **already satisfied** by the engine's group means/SDs.

### 2.3 Implementation plan

- **E-step:** unchanged in structure; item response tables become group-indexed for flagged items (the
  engine already integrates per group). No new integration dimension.
- **M-step:** for each flagged item `i`, run the existing per-item Newton **once per group** on that group's
  expected counts to update `(a_i^g, b_i^g)`; anchor items keep the pooled update. Reference group pinned
  `N(0,1)`.
- **Parameter count:** per flagged item, `+ (H−1)` for uniform DIF (`b_i^g`), `+ (H−1)` more for
  non-uniform (`a_i^g`), `H` = #groups. Impact params `μ_{g,d}, σ_{g,d}` already exist.
- **Identification:** reference group `N(0,1)`; anchor set spans and is DIF-free (support optional
  purification: iterate — refit, drop anchors whose Wald DIF is significant, refit). With simple structure
  the "≥1 anchor per testlet" rule becomes "≥1 anchor per latent dimension `d`."
- **DIF test:** Wald test `d̂_{ih}/SE` from the observed-information SEs the M-step already produces
  (cheapest); or LR by refitting with `b_i^g` constrained equal. Provide both; they agree asymptotically.
- **Minimal recovery test:** 2 groups, `I=30`, 6 anchors, impact `μ_2=0.5, σ_2=1.2`, plant uniform DIF
  `d=0.5` on 3 items and `0` on the rest. Assert: `d̂` within ±0.05 on planted items; Wald Type-I near 0.05
  on null items; and that omitting impact (`μ_2=0`) inflates DIF estimates (reproduces the paper's key
  warning).

### 2.4 Out of scope

The bifactor / testlet structure itself (general + conditionally-independent specific dimensions), the
Cholesky `C_h` cross-group covariance/correlation estimation, the graphical-model junction-tree E-step,
differential *testlet* functioning (`μ_{kh}≠0`), and polytomous link functions. Only the multiple-group
**DIF-on-simple-structure** slice is ported.

---

## 3. Debeer & Janssen (2013) — item-position effects

**Citation.** Debeer, D., & Janssen, R. (2013). Modeling Item-Position Effects Within an IRT Framework.
*Journal of Educational Measurement*, 50(2), 164–185.

### 3.1 The paper's models, exact

Base (Eq. 1), person `p`, item `i`, position `k`, difficulty `β_{ik}`:
$$
\operatorname{logit}[Y_{pik}=1]=\theta_p-\beta_{ik}.
$$

DIF-style decomposition of position (Eq. 2), `β_i` = reference-position difficulty, `δ^{β}_{ik}` = position
shift:
$$
\operatorname{logit}[Y_{pik}=1]=\theta_p-\big(\beta_i+\delta^{\beta}_{ik}\big).
$$

2PL with position effects on both parameters (Eq. 3):
$$
\operatorname{logit}[Y_{pik}=1]=\big(\alpha_i+\delta^{\alpha}_{ik}\big)\big[\theta_p-\big(\beta_i+\delta^{\beta}_{ik}\big)\big].
$$

Position-only (not item-dependent) main effect (Eq. 4):
$$
\operatorname{logit}[Y_{pik}=1]=\theta_p-\big(\beta_i+\delta^{\beta}_{k}\big).
$$

**Linear position effect on difficulty** (Eq. 5), `γ` = shared linear slope, first position = reference:
$$
\boxed{\;\operatorname{logit}[Y_{pik}=1]=\theta_p-\big[\beta_i+\gamma\,(k-1)\big]\;}
$$
`γ>0` = fatigue (harder later), `γ<0` = practice/learning. Nonlinear (quadratic/cubic/exponential) forms
allowed by replacing `(k−1)`.

**Person-specific (random) position effect** (Eq. 6), `γ_p ~ N(·,·)`, correlated with `θ_p`:
$$
\operatorname{logit}[Y_{pik}=1]=\alpha_i\big[\theta_p-\big(\beta_i+\gamma_p\,(k-1)\big)\big].
$$
Model (6) is two-dimensional; `corr(γ_p, θ_p)` is estimable. Empirically `γ ≈ .01–.24` per position/cluster,
`corr(γ_p,θ_p) < 0` (higher-ability persons less affected).

**Identification.** A reference position with zero effect (first position, since `γ·(k−1)`). Eq. 2/3 need a
per-item reference position; Eq. 4–6 share one reference across items. **Selection:** nested models by LR;
random-slope (6 vs 5) uses a **mixture-of-`χ²`** boundary test; fixed `δ`/`γ` significance by **Wald**.

### 3.2 Mapping to the engine

Position enters the linear predictor **additively** (Eq. 4/5), which is exactly a **per-cell covariate
offset**. Write the position covariate `w_{pi} = k(p,i)−1` (position of item `i` on the form person `p`
took, minus 1; or cluster-position for rotated-block designs). In the engine's easiness convention
(`b_i = −β_i`):

$$
\boxed{\;
\eta_{pi}=a_i\,\theta_{p,d(i)}+b_i-\gamma\lVert\xi_p-\zeta_i\rVert \;+\; w_{pi}\,\delta,
\qquad \delta=-\gamma_{\text{pos}}
\;}
$$

Two useful configurations, both from the paper:

- **Test-level (shared) slope `δ`** (Eq. 5): a single extra scalar. This is the minimal, recommended form
  (the paper's simulation and both applications land on the linear shared-slope model as best fit).
- **Item-level slope `δ_i`** (Eq. 2/4, "position DIF per item"): `w_{pi} δ_i`, one slope per item — the
  `δ^{β}_{ik}`/`δ^{β}_k` main-effect family. Choose the position basis in `w`: linear `(k−1)`, or dummy
  columns per position for the unstructured Eq. 4 main effect.

The offset is a **known covariate times an unknown slope** — a GLM offset with a free coefficient, needing
only the linear predictor to gain one additive term and the M-step to gain one (or `I`) coordinate.

### 3.3 Implementation plan

- **E-step:** the quadrature kernel `P_i(v)` becomes `P_{pi}(v) = logit^{-1}(a_i v + b_i − γ‖·‖ + w_{pi}δ)`.
  Because `w_{pi}` is data (person×item design), the person-independent item-table shortcut is broken **only
  when `w_{pi}` varies within an item across persons** (random/rotated orders). For a single fixed form,
  `w_{pi}=w_i` is item-constant and the existing tables still apply. Keep both paths.
- **M-step:** add `δ` (or `δ_i`) to the Newton block. Gradient contribution per cell is
  `w_{pi}·(expected residual)`; for shared `δ` it is a 1-D Newton summed over all cells, identical in form to
  the existing `γ` update.
- **Parameter count:** `+1` (shared linear), `+I` (per-item), or `+(K_pos−1)` (unstructured position dummies).
- **Identification:** reference position `k=1` gives `w=0`, pinning the offset; no extra constraint. `δ` is
  identified only if items appear at **≥2 distinct positions** across the data (overlapping/anchor items
  across forms, or randomized order) — otherwise position is confounded with item easiness. Enforce/warn.
- **Person-specific `γ_p` (Eq. 6):** *adaptation, heavier.* It is a **random slope** = a second
  simple-structure latent dimension whose "loadings" are the fixed known values `w_{pi}=(k−1)`, correlated
  with `θ`. Implementable as a 2-D correlated trait (`corr(γ_p,θ_p)` via a `2×2` population covariance) but
  requires correlated quadrature — see §5 (same missing capability as Huo's cross-dimension covariance).
  Document as the upgrade path; ship the fixed-slope offset first.
- **Minimal recovery test:** `N=1000`, `I=50` drawn per person from a pool of 75 with random order (paper's
  design), fixed `γ_pos ∈ {.010,.015,.020}`. Assert: `δ̂` recovers `−γ_pos` within ±.003; plain LSIRM (δ=0)
  overestimates difficulty by ≈ (mean position)·γ_pos (their Table 1 finding); person params ~unbiased.

### 3.4 Out of scope

Random position slope `γ_p` beyond the documented 2-D adaptation; response-contingent ("dynamic") position
models (Verguts–De Boeck, Verhelst–Glas); pairwise/sequencing effects (item-preceded-by-item); position
effects on response *time*; and speededness/omission mechanisms (the paper explicitly separates these from
position effects — omitted/not-reached items must be handled by the missing-data path, not the offset).

---

## 4. Jeon & De Boeck (2016) — generalized IRTree

**Citation.** Jeon, M., & De Boeck, P. (2016). A generalized item response tree model for psychological
assessments. *Behavior Research Methods*, 48(3), 1070–1085. https://doi.org/10.3758/s13428-015-0631-y

### 4.1 The paper's model, exact

A response with `M` observed categories is decomposed by a tree into `K` internal binary (or polytomous)
**nodes**. Node `k` for person `p`, item `i` uses its own IRT model (Eq. 1/7):
$$
\Pr\big(Y^{*}_{pik}=T_{mk}\mid\theta_{pk}\big)=g^{-1}\big(\alpha_{ik}\,\theta_{pk}+\beta_{ik}\big),
$$
with node-specific latent trait `θ_{pk}`, slope `α_{ik}`, intercept `β_{ik}`, and link `g` (logit/probit for
binary nodes; adjacent/cumulative logit for >2-branch nodes).

**Mapping matrix `T`** is `M×K`; entry `T_{mk} ∈ {0,1,…,L−1}` is the outcome required at node `k` on the
path to observed category `m`, and `NA` when node `k` is **off-path** for `m`. Observed-response likelihood
(Eq. 8):
$$
\boxed{\;
\Pr(Y_{pi}=m\mid\theta_{p1},\dots,\theta_{pK})=\prod_{k=1}^{K}\Pr\big(Y^{*}_{pik}=T_{mk}\mid\theta_{pk}\big)^{t_{mk}},
\quad t_{mk}=\begin{cases}T_{mk}, & T_{mk}\in\{0,1\}\\[2pt]0,& T_{mk}=\mathrm{NA}\end{cases}
\;}
$$

The two structural assumptions: (i) internal-node outcomes are conditionally independent given the traits;
(ii) exactly one path yields each observed category. The traits `θ_p=(θ_{p1},…,θ_{pK})' ~ N(0,Σ)`.

**Key equivalence (Eq. 9).** Model (8) **is a simple-structure `K`-dimensional IRT model** fit to the
node-expanded binary responses `Y*_p = (Y*_{p1},…,Y*_{piK})`, with **structural missingness** wherever a
node is off-path (`NA`). No cross-loadings between node-dimensions → identified by the usual simple-structure
constraints (`means 0, diagonal Σ variances 1`).

Optional refinements:
- **Node-main-effect reduction (Eq. 11–12):** `β_{ik}=β_i+δ_{βk}`, `α_{ik}=α_i+δ_{αk}` — collapses `I×K`
  node-specific parameters to `I+K`, and tests node-measurement invariance.
- **Bifactor node structure (Eq. 10):** `g^{-1}(α^{g}_{ik}θ^{g}_p+α_{ik}θ_{pk}+β_{ik})` — a general factor
  loading all nodes (within-item multidimensionality).
- **Collapse latent structure (Eq. 13):** if node traits are perfectly correlated, a single `θ_p`.

### 4.2 Mapping to the engine — pseudo-item expansion

Because Eq. 9 says the IRTree **is** a simple-structure `K`-dimensional model with structural missingness,
and the engine already does simple-structure multidim + not-presented missingness, the **entire minimal
model is a data-preprocessing layer plus interpretation — no core E-/M-step change.**

Expansion procedure (the deliverable for a binary-response engine):

1. Fix a tree and its `M×K` mapping matrix `T` (e.g., three-category "No/Perhaps/Yes" → `K=2`; four-point
   Likert → `K=3`; omit-then-respond → `K=2` with node 1 = responded/omitted).
2. For each original response `Y_{pi}=m`, emit `K` **pseudo-items**. Pseudo-item `(i,k)` gets value `T_{mk}`
   if `T_{mk}∈{0,1}`, and is **left missing** (dropped from the E-step, exactly the not-presented path) if
   `T_{mk}=NA`. Behavioral/omitted responses are just another observed category `m` with its own row of `T`
   (this is how the paper models MNAR omission: node 1 = respond/omit).
3. Assign each pseudo-item `(i,k)` to latent dimension `d = k` (simple structure: node = dimension).
   Node-specific `(a_{ik}, b_{ik})` are simply free per pseudo-item — the natural default.
4. Run the engine unchanged. Node traits’ correlations are the engine's between-dimension population
   correlations (if it estimates them; see §5) — otherwise fit diagonal and report per-node traits.

This directly delivers: partial-ordering tests of Likert scales, response-style dimensions, and
skip/omission (MNAR) modeling — all as expansions, all binary.

### 4.3 Implementation plan

- **E-step / M-step:** unchanged. The expansion produces a binary matrix with missing cells the engine
  already integrates over and updates from.
- **Preprocessing module:** `expand_irtree(Y, T) -> (Y*, dim_map)` — takes the original responses and a
  mapping matrix, returns the pseudo-item binary matrix, the per-pseudo-item dimension assignment `d=k`, and
  the missingness mask. This is the whole feature.
- **Node-main-effect reduction (Eq. 11–12):** *optional adaptation.* Parameter-tying `a_{ik}=a_i+δ_{αk}`,
  `b_{ik}=b_i+δ_{βk}` across pseudo-items of the same original item — a linear constraint in the M-step
  (shared `a_i,b_i` plus per-node offsets `δ_k`). Ship free per-pseudo-item first; add tying if parameter
  economy or invariance testing is wanted.
- **Parameter count:** free version = `K` binary items per original item, each with `(a,b)` → `2·I·K`;
  reduced version → `2·I + 2·(K−1)`.
- **Identification:** simple-structure constraints per node-dimension (`mean 0, var 1`). **LR caveat from the
  paper:** likelihoods are comparable only between trees of the **same size** (same node vector length); a
  tree that changes `K` changes the expanded data, so use AIC/BIC or refit, not a raw LR, across different
  trees.
- **Minimal recovery test:** simulate `K=2` tree (3-category), `I=24`, `N=316` (verbal-aggression sizes),
  known node-specific `(a,b)` and `corr(θ_1,θ_2)`. Assert: expand → fit recovers node params within Monte
  Carlo error and the node-trait correlation; and that treating the 3-category item as a single dichotomized
  binary loses the second node's information (sanity check on the value of the expansion).

### 4.4 Out of scope

Polytomous (>2-branch) nodes with adjacent/cumulative links (GPCM/GRM at a node) — needs a polytomous
kernel the engine does not have; keep to **binary nodes** (the paper's own primary illustrations are binary).
Bifactor node structure (Eq. 10, within-item general factor) — requires cross-loading, not simple structure.
Multiple-path trees (Böckenholt 2013) — explicitly excluded by the paper's one-path assumption. Node-specific
person **covariates** (Eq. 14) beyond what the engine already supports.

---

## 5. Huo, de la Torre, Mun, Kim, Ray, Jiao & White (2015) — hierarchical multi-unidimensional 2PL for sparse multi-group IDA

**Citation.** Huo, Y., de la Torre, J., Mun, E.-Y., Kim, S.-Y., Ray, A. E., Jiao, Y., & White, H. R. (2015).
A Hierarchical Multi-Unidimensional IRT Approach for Analyzing Sparse, Multi-Group Data for Integrative Data
Analysis. *Psychometrika*, 80(3), 834–855. https://doi.org/10.1007/s11336-014-9420-2

### 5.1 The paper's model, exact

Between-item (multi-unidimensional) 2PL for respondent `i` in group `g`, item `j` of dimension `d` (Eq. 1):
$$
P\big(X_{gij(d)}=1\mid\theta_{gi(d)},\alpha_{j(d)},\beta_{j(d)}\big)
=\frac{\exp\!\big[\alpha_{j(d)}\big(\theta_{gi(d)}-\beta_{j(d)}\big)\big]}
{1+\exp\!\big[\alpha_{j(d)}\big(\theta_{gi(d)}-\beta_{j(d)}\big)\big]},
$$
each item loads **one** dimension `d=1,…,D` (simple structure). Group-`g` likelihood (Eq. 2):
$$
L(X_g\mid\theta_g,\mu_g,\Sigma_g,\alpha,\beta)=
\prod_{d=1}^{D}\prod_{i=1}^{I}\prod_{j(d)}
\big[P_{gij(d)}\big]^{X_{gij(d)}}\big[1-P_{gij(d)}\big]^{1-X_{gij(d)}} .
$$

**Hierarchical latent structure.** `θ_{gi} ~ N(μ_g, Σ_g)` — each group has its own `D`-vector mean and
`D×D` covariance. **Anchor group** `g=G`: `μ_G = 0`, and `Σ_G` constrained to a **correlation matrix `R`**
(variances 1) — identification is on the *latent distribution*, item parameters left free. Other groups
estimate full `μ_g, Σ_g`. A **second hierarchical level** links the group means (real-data model, Eq. 5):
$$
\mu_g\sim N(\mu_H,\Sigma_H),\qquad \mu_H\sim N(0,\tau_H^2 I),
$$
so group means are random effects shrunk toward a grand mean `μ_H` — the mechanism that stabilizes small /
sparse studies.

**Estimation:** MCMC (Gibbs for `μ_g, Σ_g`; M–H for `θ_{gi}`, for `(α,β)`, and for the correlation matrix
`R` via a determinant-ratio acceptance). Sparse pooled data (≈57% missing) handled by the **"not presented"
(NP)** rule — the sampler skips missing cells and uses only observed responses (MAR assumed, justified by a
design-driven missingness pattern). Two-stage run: calibrate structural params on a reduced sample, then
score everyone with those params fixed.

### 5.2 Honest coverage assessment against `fast-mlsirm`

**Already covered by the target engine:**

- Between-item **simple-structure multidimensional 2PL** — this *is* the engine's `d(i)` loading map.
- **Multiple groups** with group-specific trait means and SDs (`μ_{gd}, σ_{gd}`).
- **Anchor-group identification** on the latent distribution (reference group `N(0,1)`, common/anchored item
  block) — the same philosophy Huo emphasizes over constraining item parameters. FIPC covers exactly this.
- **Sparse / MAR missingness via not-presented** — the engine's E-step already drops missing `(p,i)` cells;
  Huo's NP rule is the same device. Their second (robustness) simulation just confirms NP is unbiased under
  design missingness, which the engine inherits.
- **Two-stage calibrate-then-score** — the engine's scoring path (EAP/EAPsum with item params fixed) is the
  EM analog of Huo's calibration/scoring split.

**Genuinely missing (the real deliverable):**

1. **Free within-group cross-dimension covariance `Σ_g` (off-diagonals).** The engine carries per-dimension
   `σ_{gd}` but (per the MMLE design's tensor-GH note) integrates a **diagonal / separable** trait
   population; it does **not** estimate the `D(D−1)/2` correlations *among* the `D` dimensions within a
   group. Huo's whole value proposition — "associations across dimensions as auxiliary information improve
   trait estimates" (de la Torre & Patz) — depends on those off-diagonals. Adding them needs a **correlated**
   E-step (rotate GH nodes by a Cholesky of `Σ_g`, or QMC) and an M-step covariance update
   `Σ_{gd d'} = E_g[(θ_d−μ_{gd})(θ_{d'}−μ_{gd'})]` (posterior cross-moments — same shape as the existing
   variance update, extended to off-diagonals).
2. **Hierarchical linking / shrinkage of group means (`μ_g ~ N(μ_H, Σ_H)`).** The engine treats each `μ_g`
   as a **fixed** effect. Huo's second level makes them **random**, shrinking sparse studies toward `μ_H`.
   This is the distinctive IDA feature and is absent. In EM it is an **empirical-Bayes / two-level M-step**:
   after the usual `μ_g = E_g[θ]`, apply a James–Stein-style shrinkage `μ_g ← (Σ_H^{-1}+n_g Σ_g^{-1})^{-1}
   (Σ_H^{-1}μ_H + n_g Σ_g^{-1} μ̄_g)` and update `μ_H = mean_g μ_g`, `Σ_H` from the between-group spread
   (Huo's own `μ_H` prior/update, ported from the Gibbs full conditional to a moment update).
3. **Estimation-method mismatch (MCMC → marginal EM).** Not a model gap but a porting cost: correlation-matrix
   `R` sampling, Inverse-Wishart priors, and 4-parameter-Beta item priors are Bayesian conveniences; the EM
   port replaces them with the correlated-quadrature E-step (#1) and the moment/EB M-steps (#1, #2). Item
   priors, if wanted, become penalties in the M-step Newton.

**Net:** ~70% already-covered; the port is (1) free within-group `Σ_g` and (2) hierarchical mean shrinkage.

### 5.3 Implementation plan

- **E-step:** replace the separable trait weighting with a **group-specific correlated** weighting — GH nodes
  transformed by `μ_g + L_g z` where `L_g L_g' = Σ_g` (or QMC when `D ≥ 3`, which the engine already offers
  for higher dims). Not-presented handling unchanged.
- **M-step:** (a) item `(a_j,b_j)` via existing per-item Newton on expected counts (unchanged);
  (b) `μ_g, Σ_g` from posterior first/second cross-moments (variance update generalized to the full matrix);
  (c) **new** hyper-step: `μ_H, Σ_H` and the EB shrinkage of `μ_g` above. Anchor group `μ_G=0, Σ_G=R`
  (correlation) pins the metric.
- **Parameter count:** `+ G·D(D−1)/2` for free within-group correlations, `+ D` for `μ_H`, `+ D(D+1)/2` for
  `Σ_H`. Anchor group contributes the `D(D−1)/2` correlations of `R` only.
- **Identification:** anchor group `μ_G=0`, `Σ_G=R` a correlation matrix (unit variances). Sparse linkage
  requires enough **cross-study common (anchor) items** to bridge groups — the paper collapses near-duplicate
  items to raise linkage; the engine's anchor block is the mechanism, but **warn when a group's overlap with
  the anchor item set is below a threshold** (their small-study bias came exactly from thin overlap).
- **Minimal recovery test:** `G=3`, `D=5`, `N_g=1000`, off-diagonal correlations 0.4, group scalings
  `Σ_2=0.75Σ_1, Σ_3=1.25Σ_1`, means `μ_2=(.3,.4,.5,.6,.7)`, `μ_3=−μ_2` (Huo's own design). Assert: `μ_g` and
  `Σ_g` (incl. off-diagonals) recovered with small bias/RMSE; trait scores correlate `≥.98` with truth; then
  **induce their real-data 57% design-missingness** and assert item/mean bias stays small (their robustness
  result). Add a sparse small-group case to confirm the shrinkage step reduces small-study mean error vs.
  fixed-effect means.

### 5.4 Out of scope

The MCMC apparatus itself (Gibbs/M–H samplers, Inverse-Wishart / 4-Beta priors, Gelman–Rubin diagnostics,
correlation-matrix determinant-ratio sampling); higher-order IRT (a super-ordinate factor subsuming domains —
the paper explicitly contrasts its "means+covariances" model with the higher-order model); item-collapsing /
harmonization of near-duplicate items across studies (a data-curation step, not an engine feature); and
posterior-predictive model checking as implemented in the paper.

---

## Summary — feasibility per paper

| # | Paper | Core contribution ported | Feasibility |
|---|---|---|---|
| 1 | Perumean-Chaney 2013 | Structural-zero mixture on the all-zero pattern; `π` via one EM responsibility + reweighted counts; "prefer the richer model, boundary-correct the `π=0` test" | **adaptation** (small: +1 param, no new integration) |
| 2 | Jeon–Rijmen–Rabe-Hesketh 2013 | Multiple-group DIF as per-item group-specific `(a_i^g,b_i^g)` on flagged items + anchors + Wald/LR test; impact via existing group means/SDs | **adaptation** (bifactor itself out of scope; DIF slice mostly already-covered) |
| 3 | Debeer–Janssen 2013 | Linear position effect as an additive per-cell covariate offset `η += w_{pi}·δ` (shared or per-item slope) | **direct** (offset + 1-D Newton); random slope `γ_p` = adaptation |
| 4 | Jeon–De Boeck 2016 | IRTree = simple-structure `K`-dim binary model via mapping-matrix pseudo-item expansion with off-path `NA`; a preprocessing layer, no core change | **direct** (binary nodes); polytomous/bifactor nodes out of scope |
| 5 | Huo et al. 2015 | Hierarchical multi-unidim 2PL for sparse multi-group IDA | **already-covered** for ~70% (simple-structure multidim + multigroup + NP/MAR missing + anchor id); **adaptation** for the 2 genuine gaps: free within-group `Σ_g` off-diagonals, and hierarchical shrinkage of group means `μ_g ~ N(μ_H,Σ_H)` |
