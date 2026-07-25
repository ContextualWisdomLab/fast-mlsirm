# Group C — Implementation-Ready Specs for fast-mlsirm

Analyst: psychometrics implementation review of 5 papers.
Target engine: **fast-mlsirm** (Rust marginal-EM latent-space IRT; binary evaluation items;
EAP/MAP/EAPsum scoring; S-X²/l_z*/infit-outfit fit stats; item-screening pipeline; serving bundles).
Use case: calibrating **LLM-as-a-Judge** outputs (each judge verdict = one binary item response).

Relevant existing modules (verified in tree):
- Rust core `crates/mlsirm-core/src/`: `fitstats.rs`, `scoring.rs`, `marginal.rs`, `mmle.rs`, `nodes.rs`, `quadrature.rs`.
- Python `python/fast_mlsirm/`: `fitstats.py`, `diagnostics.py`, `scoring`(in core), `serving.py`, `report.py`, `linking.py`, `test_design.py`, `inference.py`, `simulation.py`, `io.py`.

Feasibility legend: **direct** (fits current binary/marginal engine), **adaptation** (needs new
subgroup/keying inputs but implementable), **superseded** (engine already covers it), **document-only**
(needs data the engine does not have — polytomous options, keyed reversals, human-rater panels).

---

## 1. Williamson, Xi & Breyer (2012) — Automated-Scoring Evaluation Framework  → **direct**

**Full citation.** Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A Framework for Evaluation and
Use of Automated Scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13.
(National Council on Measurement in Education.)

**Core contribution.** An ETS operational framework (built around e-rater) with *conjunctive*
acceptance criteria for approving an automated scorer to run alongside human scoring. For our purpose the
"automated score" = LLM-judge verdict, the "human score" = gold human label. Any single criterion failing
flags the item/task as a substantive concern.

### 1.1 Exact statistics and thresholds

All criteria are **conjunctive** and must be computed on a **held-out set** (not the data used to fit the
judge/calibration model); for task-generalization, no task overlap between fit and eval sets.

| # | Statistic | Formula | Threshold | Notes |
|---|-----------|---------|-----------|-------|
| A | Quadratic-weighted kappa (QWK), auto vs human | `κ_w = 1 − (Σ w_ij O_ij)/(Σ w_ij E_ij)`, quadratic weights `w_ij = (i−j)²/(K−1)²`, `O` = observed joint prop., `E` = product of marginals (Fleiss & Cohen 1973) | **≥ .70** (auto rounded normally to the human scale), on generally-normal score distributions | "Tipping point where signal outweighs noise"; ~half of human-score variance explained. |
| B | Pearson product–moment r, auto vs human | standard `r` on **unrounded** auto scores | **≥ .70** | Same variance-accounted rationale. Differs from A because A rounds, B does not. |
| C | Exact agreement %, and exact+adjacent (±1) agreement % | proportion equal / within 1 point | **Reported only — NOT an acceptance criterion** | Rejected as a gate due to scale dependence (higher by chance on a 4-pt than 6-pt scale) and base-rate sensitivity. Report for lay readers. |
| D | Degradation from human–human agreement | `Δ = (human–human agreement) − (auto–human agreement)`, in **either** QWK or r | **auto–human may not be > .10 lower** than human–human | Requires a human–human baseline (double-scored subset) as a precursor. Borderline exception noted (e.g. auto–human .69 vs human–human .71 accepted). Auto may legitimately exceed human–human. |
| E | Standardized mean score difference (SMD) | `SMD = (M_auto − M_human) / SD_human` (standardized on the **human** score distribution) | **\|SMD\| ≤ .15** (overall/task level) | Guards against differential scaling / off-center distributions. For a regression-fit scorer, SMD is rarely flagged on the fit set → must use held-out data. |
| F | Subgroup SMD (fairness) | same as E, computed within each subgroup of interest | **\|SMD\| ≤ .10** (stricter than overall) | Applied to every relevant subgroup (Ramineni, Williamson & Weng 2011). |
| G | Discrepancy threshold for human adjudication | \|auto − human\| ≥ τ → route to another human | Program-set: **GRE τ = 0.5** ("exact agreement"); **TOEFL τ = 1.5** | Policy knob, not a pass/fail metric; tuned to legacy human double-scoring policy. |
| H | Human-intervention filters (advisory-input screen) | rule-based flags | flag excessive length/brevity, repetition, "too many problems", off-topic | Route flagged responses to human; keep as a config-driven pre-filter. |

Supporting (framework, not single-number gates): human scoring process/inter- & intra-rater reliability
review (prerequisite); within-test and external-criterion relationship comparisons (auto vs human);
generalizability G/Phi coefficients across tasks/forms and prediction of human scores on an alternate form;
impact-on-decision-accuracy and subgroup checks on agreement, generalizability, prediction, and decisions.

### 1.2 Binary-item reductions (LLM-judge case; K = 2)

The scale is 2 points (fail/pass = 0/1), so:
- **A (QWK) collapses to Cohen's unweighted kappa** — with `K=2`, `w_00=w_11=0`, `w_01=w_10=1`, so
  `κ_w = κ = (p_o − p_e)/(1 − p_e)`. Threshold **≥ .70** stands. (Note: kappa is base-rate sensitive
  when pass-rate is extreme; report the 2×2 table alongside.)
- **B (Pearson r) collapses to the phi coefficient** on the 2×2 table. Threshold **≥ .70**.
- **C adjacent agreement is degenerate** (with 2 categories, exact+adjacent = 100%); report **exact
  agreement = accuracy** only.
- **E SMD** `= (p_judge − p_human)/sqrt(p_human(1−p_human))`. Threshold **|SMD| ≤ .15**.
- **D degradation** needs a human–human κ from a double-labeled subset; **|Δκ| ≤ .10**.
- **G** with binary scores reduces to "disagree → adjudicate" (τ between 0 and 1).

### 1.3 Implementation plan

- **Module:** new `python/fast_mlsirm/validation.py` ("machine-scoring validation metrics"), sibling to
  `diagnostics.py`. Optionally push the hot kernels (kappa, phi, SMD over large N) to a new
  `crates/mlsirm-core/src/validation.rs` mirroring the `fitstats.rs` compute-in-Rust/parity-in-NumPy pattern.
- **Formulas to code:**
  1. `cohen_kappa(judge, human)` and general `quadratic_weighted_kappa(a, b, K)` (weights above).
  2. `pearson_r` / `phi` on paired vectors.
  3. `smd(auto, human)` per §1.1-E and §1.2.
  4. `degradation(auto_human_stat, human_human_stat)` → returns Δ and pass flag at .10.
  5. `subgroup_smd(auto, human, group_id)` → per-group SMD, pass flag at .10.
  6. `agreement_report(...)` bundling exact-agreement %, 2×2 table (reported, non-gating).
  7. A `ValidationVerdict` dataclass with the conjunctive PASS/FLAG rollup + which criteria failed;
     surface it in `report.py` and attach to the `serving.py` bundle as a calibration gate.
- **Inputs:** paired `(judge_label, human_label)` arrays + optional `subgroup_id`; a double-labeled subset
  for the human–human baseline (criterion D); held-out flag.
- **Minimal test:** hard-code a 2×2 confusion table with a hand-computed kappa/phi/SMD (e.g. from a fixed
  contingency matrix) and `assert` the functions reproduce them to 1e-9; one degradation case that must FLAG.

### 1.4 Out of scope / caveats

Criterion D **requires human–human double-scored data** (a rater panel); without it, degradation cannot be
computed — degrade gracefully (report A/B/E/F, mark D "N/A: no human–human baseline"). The G/Phi
generalizability and external-criterion analyses need multi-task/multi-form or external-variable data and
are framework guidance, not a single coded metric. Adjacent agreement is not meaningful for binary items.

---

## 2. Ferrando, Lorenzo-Seva & Chico (2009) — FA Procedure for Response Bias  → **document-only** (acquiescence analog = adaptation)

**Full citation.** Ferrando, P. J., Lorenzo-Seva, U., & Chico, E. (2009). A General Factor-Analytic
Procedure for Assessing Response Bias in Questionnaire Measures. *Structural Equation Modeling, 16*(2),
364–381. DOI: 10.1080/10705510902751374.

**Core contribution.** A semirestricted **tridimensional** factor-analytic model that simultaneously
separates **content (θ₁), acquiescence (θ₂), and social desirability / SD (θ₃)** from questionnaire items,
with a three-stage non-iterative calibration and factor-score estimation.

### 2.1 Exact model, anchoring, and steps

Structural model per content item (z-metric, three **uncorrelated** factors), Eq. 2:
```
X_ij = φ_j1·θ_i1 + φ_j2·θ_i2 + φ_j3·θ_i3 + ε_ij
```
SD-marker items (part of a lie/control scale) are factorially simple, Eq. 3: `X_ik = φ_k3·θ_i3 + ε_ik`.
Two key structural assumptions: (i) content, acquiescence, SD mutually independent; (ii) acquiescence does
**not** operate on near-pure SD items. Binary case = MIRT 2-parameter normal-ogive on the **tetrachoric**
correlation matrix; graded = polychoric; continuous = product-moment.

**Three-stage sequential calibration** (one factor per stage; general FA engine = **Minimum Rank Factor
Analysis, MRFA**, which also yields item error variances and %-common-variance per factor):
1. **SD (θ₃) via instrumental-variable estimation** (Hägglund 1982): need **≥ 3 SD markers** (4 recommended).
   Take one marker as pivot/proxy for θ₃, the remaining `m−1` markers as instruments;
   `φ̂'_j3 = (r_k′ r_k)^{-1} (r_j′ r_k)` (Eq. 17). Reproduce and subtract → first residual matrix.
2. **Acquiescence (θ₂)** from the first residual matrix by the **modified-centroid** formula, using the
   **weak balance assumption** (sum of content loadings over a balanced +/− keyed subset ≈ 0). For a
   balanced-subset item `j`: `φ̂_j2 = (Σ_g r*_jg − s²_j) / sqrt(Σ_j Σ_g r*_jg − Σ_j s²_j)` (Eq. 18);
   analogous Eq. 19 for non-balanced items. Reproduce and subtract → second residual matrix.
3. **Content (θ₁)** = one-common-factor (Spearman) MRFA on the second residual matrix.

**Anchoring.** SD is anchored by the **marker items** (must be positively identified — large SD loadings,
small content loadings). Acquiescence is anchored by the **partial balance** of positive/negative keyed
items (needs both keying directions). Fit is judged non-inferentially (RMSR of residuals, residual-distribution
shape) — no χ² test. Factor scores: **EAP** (nonlinear/binary/graded) or **Bartlett ML** (continuous), with
posterior SD `PSD = sqrt(E(θ²|x) − θ̂²)` (Eq. 21) and marginal reliability `ρ = 1 − mean(PSD²)` (Eq. 22).

### 2.2 Why this is document-only for fast-mlsirm

The method's inputs do not exist in the LLM-judge/binary calibration setting:
- Needs a **partially balanced item set** (positively *and* negatively keyed items). Judge verdicts have no
  keying reversal — there is no "disagree" polarity to cancel content loadings, so acquiescence is
  unidentified by this route.
- Needs a dedicated **multi-item SD/lie marker scale** administered alongside — absent here.
- Built on **correlation-matrix FA + MRFA + IV estimation**, an entirely different estimation stack from
  fast-mlsirm's per-response marginal EM; it is a calibration *replacement*, not an add-on.

### 2.3 Salvageable adaptation (optional, small)

The *concept* — that a judge may have a content-independent "yes-tendency" (leniency/acquiescence) or a
"desirability" pull — is worth a lightweight diagnostic, but **not via this FA machinery**. A defensible
analog inside the current engine: after MMLE, report each judge's **base pass-rate residual** (observed
pass-rate minus model-expected pass-rate marginalized over θ) as a leniency index, and, if paired
positive/negative-framed prompt variants exist, a keying-direction contrast. Place in `diagnostics.py`.
Do **not** attempt the tridimensional FA — it needs data we do not collect. `# ponytail: leniency = one
residual, not a 3-factor MRFA; upgrade only if keyed-pair prompts are added.`

### 2.4 Out of scope

Full procedure requires: keyed (reversed) items, an SD marker scale, tetrachoric/polychoric matrices with
smoothing (Devlin et al. 1975/1981), and MRFA (FACTOR software). None are in scope for binary judge calibration.

---

## 3. Wolkowitz & Skorupski (2013) — MCM Imputation of MC Response Options  → **superseded**

**Full citation.** Wolkowitz, A. A., & Skorupski, W. P. (2013). A Method for Imputing Response Options for
Missing Data on Multiple-Choice Assessments. *Educational and Psychological Measurement, 73*(6), 1036–1053.
DOI: 10.1177/0013164413497016.

**Core contribution / exact method.** Uses Thissen & Steinberg's (1984) **Multiple-Choice Model (MCM)** — a
`(3n−1)`-parameter logistic nominal-type model over all `n` options — to **multiply-impute the actual chosen
option** (A/B/C/D/E) for missing responses, so that classical item statistics (p-values, item–total r)
become robust. MCM (Eq. 1):
```
P(u_ij = k | θ_i) = [ d_jk · exp(a_j0 θ_i + b_j0) + exp(a_jk θ_i + b_jk) ] / Σ_{h=0}^{m_j} exp(a_jh θ_i + b_jh)
```
constraints `Σ a_jh = 0`, `Σ b_jh = 0`, `Σ_{k≥1} d_jk = 1`; the `d` params + option-0 term form the
"guessing/don't-know" curve, split from the "intentional" curve (Eq. 4). **MI procedure:** (1) calibrate MCM
on the incomplete data (MULTILOG, listwise-ignores missing); (2) compute per-option probabilities for each
missing cell via Eq. 4; (3) Monte-Carlo draw `X~U(0,1)` to assign an option; (4) recompute statistics;
repeat **m = 100** times; bias = mean(estimate) − true, efficiency = SD across imputations. Result: under
**MNAR**, case-deletion overestimated p by ~+.04 (up to .15 for mid-difficulty items); MI-with-MCM shrank
bias to <.01. Under MCAR/MAR both methods were ~unbiased.

### 3.1 Honest note: does marginal-ML MAR handling supersede it?

**Yes, for calibration purposes.** The paper itself names ML and MI as the two recommended modern
approaches (Schafer & Graham 2002); fast-mlsirm already uses the **ML branch**: marginal maximum likelihood
integrates over unobserved responses so that **MCAR and MAR missingness is ignorable and needs no
imputation** — item parameters are estimated consistently from observed cells. So the paper's own goal
("more robust item statistics under MCAR/MAR") is met directly by MMLE with no imputation step.

Two honest boundaries:
- **MNAR** is not solved by either MMLE or MCM in general; MCM only appears to fix MNAR here because the
  missingness was simulated *from the MCM intentional-curve itself* (the imputation model equals the
  missingness model — a best case). Marginal ML under genuine MNAR is biased too; the remedy is an explicit
  missingness/selection model, not option imputation.
- MCM imputes the **specific distractor** (A/B/C/D/E). fast-mlsirm items are **binary** (0/1) with no
  distractor structure, so there is nothing to impute at the option level — the (3n−1) machinery has no
  target. If a missing binary verdict must be filled for reporting completeness, that is a 1-line posterior
  draw `Bernoulli(P(x=1|θ̂))`, not the MCM.

### 3.2 Implementation plan

**None recommended for the core.** MMLE already handles the intended MAR/MCAR case. If a "completed-matrix
for reporting" convenience is ever wanted, add a `impute_missing()` helper in `diagnostics.py` that draws
`Bernoulli(predict_proba)` per missing cell over `m` replications and reports across-imputation SD — reusing
existing `predict_proba`. `# ponytail: MMLE integrates out MAR; skip imputation unless a filled matrix is a
hard reporting requirement.`

### 3.3 Out of scope

MCM requires **polytomous multiple-choice option data** and MULTILOG-style nominal calibration; incompatible
with binary judge items. Its MNAR success is an artifact of matched simulate/impute models — do not cite it
as an MNAR guarantee.

---

## 4. Makransky & Glas (2013) — DIF via Group-Specific Item Parameters (CAT)  → **direct / adaptation**

**Full citation (previously unknown #4).** Makransky, G., & Glas, C. A. W. (2013). Modeling differential
item functioning with group-specific item parameters: A computerized adaptive testing application.
*Measurement, 46*(9), 3228–3237. Elsevier. DOI: 10.1016/j.measurement.2013.06.020.

**Core contribution.** A measurement-invariance / DIF workflow in a **2-PL, MML-estimated** IRT model:
detect DIF with the **Lagrange-Multiplier (LM)** and **Wald** statistics, then, instead of deleting DIF
items, split each into **"virtual items"** with **group-specific parameters** so DIF items still contribute
information while subgroups stay on a common scale. Crucially, the LM statistic is coded with an
**observed-response indicator**, so it works for **incomplete / CAT** designs — exactly fast-mlsirm's
marginal + MAR setting.

### 4.1 Exact statistics and thresholds

2-PL (Eq. 1): `P_i(θ) = 1/(1 + exp(−a_i(θ − b_i)))`.

**LM statistic.** Split respondents into subgroups `g = 1…G` (focal/reference, or score-level groups for
model-fit). Per item `i`, subgroup mean observed score (Eq. 2):
```
S_ig = (1/N_g) Σ_{n in g} b_ni · X_ni
```
where `X_ni` = observed response (0/1) or dummy if unobserved, and `b_ni = 1` if observed else `0` (this
indicator is what makes it CAT/missing-data safe). Compare `S_ig` to its posterior expectation `E(S_ig)`;
square the differences and weight by their covariance matrix. **LM ~ χ² with G−1 df.** Effect size (Eq. 3):
```
d_ig = max_g |S_ig − E(S_ig)|
```
on the observed-score scale `0…m_i` (here `m_i = 1`, binary). **Threshold: `d_ig > 0.10` = more than minor
model violation** (rule of thumb, Glas 1998/2010) — valid specifically for **dichotomous** items. Because LM
power grows with N, **prefer effect size over p-value**.

**Wald statistic.** Directly contrasts the MML item-parameter estimates (a, b) across subgroups; supports
scatter-plots of subgroup a's and b's for eyeballing misfit. Item flagged if **Wald significant** OR
**LM `d_ig > 0.10`**.

**Iterative purification (screening pipeline).** Estimate concurrently → flag highest-DIF items via Wald/LM →
assign those **group-specific (virtual-item) parameters** → re-run Wald/LM on the remaining common items →
repeat until no common items show DIF. Then a final concurrent LM check (item-response-curve form + local
independence) confirms all virtual + common items fit one model → subgroup person scores are comparable.

### 4.2 Implementation plan (strong fit)

- **Fit-statistics module — `fitstats.rs` / `fitstats.py`:** add `lm_dif(...)` alongside the existing S-X²/
  l_z*. The LM machinery (subgroup observed means vs posterior expectations over the quadrature grid,
  covariance-weighted quadratic form) reuses the same `(theta, xi)` node/weight grid already built for S-X²;
  the `b_ni` observed-indicator maps directly onto fast-mlsirm's existing response mask. Emit both χ²/p and
  `d_ig`, with the **0.10** binary threshold as the default flag.
- **Item-screening pipeline — `test_design.py` / `diagnostics.py`:** implement the iterative purification loop
  (flag → assign group-specific params → re-fit → repeat). This is the "judge-DIF" screen: subgroups =
  prompt category, evaluated-model family, language, or content demographic slice; flags judges/items whose
  difficulty/discrimination differs by slice.
- **Optional model extension (adaptation):** support **group-specific item parameters** in `marginal.rs`/
  `mmle.rs` (virtual items = duplicate an item's `(alpha, b, zeta)` per group) so DIF items are retained
  rather than dropped from the calibration/serving bank.
- **Wald:** add `wald_dif(params_g1, params_g2, cov)` in `fitstats.py` from the already-available MML
  parameter covariance.
- **Minimal test:** simulate 2 subgroups from `simulation.py`, inject a known b-shift into one item, assert
  `lm_dif` flags exactly that item at `d_ig > 0.10` and leaves clean items unflagged; assert `b_ni`-masked
  (CAT-style) input reproduces the full-data LM on the observed cells.

### 4.3 Out of scope / caveats

The `d_ig > 0.10` cutoff is calibrated for **dichotomous** items (effect sizes are category-weighted);
re-derive if ever extended to polytomous. Needs a **subgroup label** per response (new input column). Virtual
/ group-specific parameters require the model-extension above before DIF items can be *retained* — without
it, the pipeline can still *detect and drop* (the classic approach). Group-specific scaling assumes an
anchor set of DIF-free common items for identification.

---

## 5. Joubert, Inceoglu, Bartram, Dowdeswell & Lin (2015) — Forced-Choice vs Likert Equivalence  → **document-only**

**Full citation (previously unknown #5).** Joubert, T., Inceoglu, I., Bartram, D., Dowdeswell, K., & Lin, Y.
(2015). A Comparison of the Psychometric Properties of the Forced Choice and Likert Scale Versions of a
Personality Instrument. *International Journal of Selection and Assessment, 23*(1), 92–97. (SHL Group / CEB.)

**Core contribution / method.** An empirical equivalence study (N = 349 SA training delegates) comparing a
**Thurstonian-IRT-scored forced-choice** questionnaire (OPQ32r, 104 triplet blocks) against a **classically
scored single-stimulus 5-point Likert** version (OPQ32n, 230 items), across 32 personality scales. The
method of interest is **Brown & Maydeu-Olivares (2011) Thurstonian IRT** — modeling "most/least like me"
block choices via the Law of Comparative Judgement to recover **normative** trait scores from forced-choice
(ipsative) data, removing ipsative distortion while controlling uniform response biases.

**Statistics/thresholds reported** (equivalence evidence, not thresholds to code into our engine):
- Reliability: Cronbach's α (OPQ32n, mean .83) vs IRT **empirical reliability** from the test-information
  function (OPQ32r, mean .83).
- **Profile similarity** = per-person correlation across the 32 scale scores (median r = .73; 63% ≥ .70;
  86% ≥ .60).
- **Profile distance** = mean of standardized-score differences across scales (96% within 0.5 z ≈ 1 sten).
- Scale intercorrelation patterns compared (both ~70% within ±0.20); same-scale n-vs-r correlations .50–.84
  (median .73).
- **Covariance-structure equivalence via SEM** (EQS): CFI = .967, RMSEA = .039, SRMR = .054 (χ² = 753.9,
  df = 496, sample-size-inflated, discounted).

### 5.1 Why document-only

- The scoring model is **Thurstonian IRT over forced-choice blocks** (triplets/quads of paired comparisons),
  a fundamentally different data structure (rank/comparative choices, multidimensional) from fast-mlsirm's
  **independent binary items**. There is no forced-choice block structure in LLM-judge verdicts.
- The paper's outputs are **polytomous/continuous 32-scale personality profiles**; our engine calibrates
  binary evaluation items on a (typically) low-dimensional latent space.
- Its equivalence metrics (α vs IRT empirical reliability, SEM covariance-structure invariance, profile
  similarity/distance) presuppose **two parallel instruments and multi-scale profiles** — absent here.

### 5.2 Salvageable ideas (no core work)

Two concepts transfer as *reporting conventions*, not new estimators:
- **IRT empirical reliability from the information function** (`ρ = 1 − mean(PSD²)` / info-based) — fast-mlsirm
  already produces posterior SDs in scoring, so an empirical-reliability line is a trivial `report.py`
  addition if not already present.
- **Cross-scorer profile similarity / distance** — if two judges (or judge vs human panel) produce vectors of
  per-item θ or pass-rates, a per-unit correlation (similarity) and standardized-difference (distance) mirror
  §1 (Williamson) agreement metrics; fold into the `validation.py` from Paper 1 rather than a new module.

### 5.3 Out of scope

Thurstonian forced-choice IRT, ipsative-data handling, multidimensional 32-scale profiles, and two-instrument
SEM invariance — all require data structures the binary engine does not model. No implementation.

---

## Cross-paper build priority for fast-mlsirm

1. **Paper 1 (Williamson)** → new `validation.py` machine-scoring gate (kappa/phi/SMD/degradation/subgroup),
   wired into `report.py` + `serving.py`. Highest value, direct fit. Also absorbs Paper 5's profile
   similarity/distance and empirical-reliability reporting ideas.
2. **Paper 4 (Makransky-Glas)** → `lm_dif`/`wald_dif` in `fitstats.*` + iterative purification in the
   screening pipeline; optional group-specific (virtual) item parameters in `marginal.rs`/`mmle.rs`. Direct
   fit; the `b_ni` observed-indicator already matches our MAR handling.
3. **Paper 3 (Wolkowitz)** → no core work; MMLE already covers MAR/MCAR. Optional `Bernoulli` fill helper only.
4. **Paper 2 (Ferrando)** → no core work; optional lightweight leniency-residual diagnostic. Full FA method out
   of scope (needs keyed items + SD markers).
5. **Paper 5 (Joubert)** → no core work; reporting-convention ideas folded into #1.
