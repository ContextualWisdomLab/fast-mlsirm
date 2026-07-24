# GPCM / Nominal Polytomous Kernel — Implementation Spec

_Synthesized by the `gpcm-design` workflow (study → 3 designs → judge → merge). The base is a unified scoring-function softmax cell nesting binary 2PL, GPCM (Muraki), and nominal (Bock), reusing `eta_at_kind` and preserving binary bit-parity via a `ResponseModel::Bernoulli` branch._

## Parametrization

UNIFIED SCORING-FUNCTION SOFTMAX CELL that nests binary 2PL, GPCM, and nominal in one kernel and REUSES `eta_at_kind` (marginal.rs L309-342) verbatim.

Base kernel (unchanged): call `eta_at_kind` with a ZERO `b` slice so it returns base_i(theta,x) = a_i*theta + I_i(x), where a_i = exp(alpha[i]) if free_alpha else 1, and I_i is the existing interaction term (Distance = -exp(tau)*||x-zeta_i||, Inner = +dot(zeta_i,x), None = 0). This scoring-function form (from the `correctness` and `gpu-first` designs) is preferred over `minimal-diff`'s per-category-slope-plus-isolated-space form because base bundles slope+space exactly as `eta_at_kind` already returns it, so nominal needs NO decomposition of the space term (removes the one friction in the base design).

Category logit (baseline category 0 pinned), k = 0..K-1:
  psi_{i0} = 0
  psi_{ik} = s_{ik} * base_i(theta,x) + c_{ik}
  P_{ik} = softmax_k(psi) = exp(psi_{ik}) / sum_h exp(psi_{ih})

Two links via ResponseModel:
- GPCM (Muraki 1992; PCM when free_alpha=false on Mlsrm/Ulsrm): s_{ik} = k FIXED (scores 0..K-1, not estimated). Free = { alpha_i (shared slope), c_{i1..K-1} (additive intercepts), zeta_i, global tau }. Steps NOT order-constrained -> no inequalities. Store ADDITIVE intercepts c_{ik} internally (not cumulative Muraki steps): the intercept gradient becomes the plain residual g_{c,m}=resid_m with no suffix-sum, and GPCM+nominal share one intercept path. Report Muraki step/difficulty via b_{i,k}=c_{i,k-1}-c_{i,k} (pure reporting reparametrization).
- NOMINAL (Bock 1972, scoring-function / operational rank-1 as in mirt & flexMIRT): pin a_i=1 (free_alpha forced false via model_exec_flags), free scoring s_{ik} and intercepts c_{ik} for k=1..K-1, baseline s_{i0}=c_{i0}=0. Fully-free per-category slopes a_{ik} (true multidimensional Bock) is a storage-only future extension; the gradient shape below is unchanged.

Binary 2PL is exactly K=2, s=[0,1], c=[0,b_i]: psi_1=base+b_i=current eta, and since logsigmoid(eta)-logsigmoid(-eta)=eta, dlp[0]=eta is a FREE parity check.

IDENTIFICATION: baseline c_{i0}=s_{i0}=0 fixes softmax translation. GPCM: scores fixed, alpha_i identified by N(0,1) theta prior + lambda_alpha, space by tau + PCA alignment (pca_align L1447-1547 UNCHANGED). Nominal: freeing both a_i and s_{ik} is non-identified (both scale base) -> pin a_i=1; label-switching mitigated by baseline-category identification + lambda penalties + ordered init s_k=k. Ragged K_i via global K=max K_i + per-item n_cat_i; slots k>=K_i get psi=-1e30 (prob underflows to 0), fixed K stride.

CRITICAL MODELING DECISION (needs maintainer sign-off, not a free lunch): because P is divide-by-total, a category-CONSTANT additive term cancels in the softmax and has identically zero gradient (sum_k resid_k=0, the `minimal-diff` insight). Therefore the latent-space interaction I_i(x) MUST enter psi_k scaled by s_k -- category k feels the LSIRM distance s_k-fold ("distance felt k-fold"). Forced by identification, not optional. Consequence: the latent-space axis is category-ORDERED/monotone, so polytomous-LSIRM "nominal" is nominal only on the theta-slopes, not on the space dimension. Novel model, no external oracle; validate internally (NumPy mirror) + recovery sim only.

## Likelihood integration (E-step cell)

The category axis is fully consumed INSIDE table-build and the cell decomposition; the theta/xi quadrature (person_pass L468-504, GPU lp_pass/nbar_pass) is category-agnostic and BYTE-UNCHANGED because l_buf[d][t][x] still holds ONE scalar log-lik per (d,t,x) (item conditional-independence within a dimension preserved; each item contributes its observed category's log-prob).

TABLES REPRESENTATION -- adopt `gpu-first`'s dlp, not `minimal-diff`'s full-K logp table (fewer hot-loop reads + cleaner binary parity check): replace Tables{logp1,logp0} with Tables{ dlp, lp0, c0, n_cat }:
  dlp[k-1] = logP_{ik} - logP_{i0} = psi_{ik}   (K-1 columns; logP_k-logP_0 = psi_k - psi_0 = psi_k)
  lp0      = logP_{i0}                            (exact role of old logp0)
  c0[(s*n_dims+d)*cell+..] = sum_{i in d} lp0[i]  ("everyone in category 0" baseline; same shape/role as today's all-fail baseline)
dlp layout: ((s*n_items+i)*(K-1)+(k-1))*cell + t*n_x + x. lp0/c0 unchanged.

BINARY BIT-PARITY RESOLUTION (resolves the tension both `correctness` and `gpu-first` flagged): build_tables_offset branches on ResponseModel, NOT a byte-identical-softmax-at-K=2 claim.
  - ResponseModel::Bernoulli (DEFAULT, dominant binary/ZI hot path): fill via the EXISTING log_sigmoid -- dlp[0]=logsigmoid(eta)-logsigmoid(-eta), lp0=logsigmoid(-eta). person_pass previously computed `logp1[i]-logp0[i]` inline; hoisting that exact subtraction into dlp is bit-identical (same IEEE operands). => CPU binary output stays bit-for-bit; the regression gate is met by NOT touching the arithmetic.
  - ResponseModel::Gpcm/Nominal: fill via HOST f64 max-subtract K-way softmax -- psi_k for k=0..K-1 (psi via eta_at_kind with b=0), m=max_c psi_c, logZ=m+ln(sum exp(psi_c-m)), lp0=-logZ, dlp[k-1]=psi_k, c0 += lp0.
So ONE reduction/GPU path serves K=2 and K>2 (no duplicated WGSL / person_pass) while the binary arithmetic path is preserved. GPCM-with-K=2 (softmax path) is validated to reduce to Bernoulli within relative ~1e-12 as a SEPARATE reduction-correctness test -- exercises the softmax path without risking the default.

PERSON_PASS decomposition (marginal.rs L453-467; only these lines change) -- the c0 + sparse-correction trick and the whole performance argument survive verbatim:
  l_buf[d] = c0[s][d]                                  // everyone in category 0
  for i in miss[p]:            l_buf[d] -= lp0[i]      // remove missing from baseline (UNCHANGED, k=0 table)
  for (i,k) in resp[p], k>=1:  l_buf[d] += dlp[i][k-1] // swap cat0->cat_k, ONE read (dlp already = logP_k-logP_0)
Observed category-0 responses are never in the response list, cost nothing, stay pooled in c0. K=2: dlp[0]=eta reproduces `+= logp1-logp0` byte-for-byte. The logsumexp reductions over t (L468-484) and x (L485-504), the person log-marginal, and the ZI mixture (zi_mix L510-520) are untouched. GPU cell_l (L83-101) mirrors this line-for-line.

index_responses (L419-433) + ResponseIndex (L414): pos category-tagged -- keep `pos: Vec<Vec<usize>>` item ids plus parallel `pos_cat: Vec<Vec<u8>>`; push (i,cat) only for observed cat>=1; miss unchanged. ZI "structural zero" (L1819) = pos[p].is_empty() STILL holds (every observed response is category 0).

## M-step gradient

Expected complete-data multinomial objective and gradient collapse to a category residual + score-weighted residual that drop into the existing m_step_items shell.

Counts per node: n = nbar - mbar (observed persons, dimension-pooled, UNCHANGED); r_k = rbar_k (expected cat-k count, k=1..K-1); r_0 = n - sum_{k>=1} r_k implicit (as binary r_0=n-r). item_q objective (L1067):
  q += sum_{k=0..K-1} r_k * logP_k = n*lp0 + sum_{k>=1} r_k*dlp[k-1]      // dlp[k-1]=psi_k

CATEGORY RESIDUAL VECTOR (generalizes scalar resid=r-n*prob at L1165; sum_k resid_k=0):
  resid_k = r_k - n*P_k,   k=1..K-1
For any parameter phi (psi_0=0 => dpsi_0/dphi=0):  dq/dphi = sum_{k>=1} resid_k * dpsi_{ik}/dphi

Per coordinate, psi_k = s_k*base + c_k:
- Intercepts (K-1), dpsi_h/dc_m=[h==m]:  g_{c,m} = resid_m.  (K=2: g_{c,1}=r-n*prob = current g_b L1167. Diagonal, one residual per intercept -- why additive c_k beats Muraki-step storage.)
- Base coords (alpha, zeta, tau, covariate), dpsi_k/dbase=s_k:  SCORE-WEIGHTED RESIDUAL R = sum_{k>=1} s_k*resid_k; then
    g_alpha = R*(a*theta)         (SAME deta as L1170)
    g_zeta[j] = R*deta_zeta[j]    (SAME deta_zeta L1176-1185: Distance gamma*(x_j-zeta_j)/dist, Inner x_j)
    g_tau (m_step_tau L1303-1321) uses R in place of scalar resid
  GPCM s_k=k => R = sum_k k*resid_k; nominal R = sum_k s_k*resid_k.
- Nominal free scoring s_m, dpsi_h/ds_m=[h==m]*base:  g_{s,m} = resid_m*base.

COVARIATE OFFSET CORRECTNESS FIX (from `correctness`, a real K>2 bug): the ItemCovariate offset currently added flat to eta (build_tables L388, m_step_delta L1388-1425) CANCELS in the softmax if added equally to every psi_k. It MUST fold into `base` (psi_k = s_k*(base + w*delta) + c_k). Then g_delta = R*w, I_delta = V*w^2. This scales the covariate effect by s_k -- the only correct divide-by-total form; a fully category-specific covariate effect is a future extension.

DIAGONAL FISHER PRECONDITIONER (keep per-coordinate, reuse the whole m_step shell):
  V = n*Var_P(s),  Var_P(s) = sum_k P_k s_k^2 - (sum_k P_k s_k)^2      // replaces n*prob*(1-prob) L1166
  i_alpha=V*(a*theta)^2 ; i_zeta[j]=V*deta_zeta[j]^2 ; i_tau=V*(gamma*dist)^2 ; i_delta=V*w^2
  i_{c,m}=n*P_m*(1-P_m)                                                // (K=2, s in {0,1}: V=n*P_1(1-P_1))
Dropped intercept off-diagonal cross-terms (-n*P_m*P_l): generalized-EM ascent + the existing 30-step Armijo backtrack on item_q (L1215-1236) backstop it. Upgrade to a (K-1)x(K-1) Newton block on intercepts ONLY if the M-step measurably stalls at large K. The damped step d=g/(I+lambda) (L1201-1205), slope check, alpha clamp [-6,3] (L1228) UNCHANGED; d_b becomes a length-(K-1) vector.

GUARD (verified hazard -- do NOT change): keep `if n<=0.0 && r<=0.0 continue` at marginal.rs L1050/1138/1306/1406 and oakes.rs L164; generalize to poly as `if n<=0.0 && r_k.iter().all(|&r| r<=0.0) continue`. `minimal-diff`'s `if n<=0 continue` drops n~0,r>0 nodes and perturbs even the binary FP trajectory -- rejected.

init (L1756-1767): additive intercepts c_{ik} from empirical log(p_k/p_0) (cumulative-category proportions); nominal scoring s_k init=k. n_free_parameters (marginal.rs L157, NOT lib.rs): per_item +(K-2) GPCM; nominal +(K-1) scoring +(K-1) intercept minus the freed alpha.

## GPU / wgpu plan

GPU-FIRST DISCIPLINE (strongest parity idea, from `gpu-first`): the K-way softmax runs on the HOST in f64 inside build_tables_offset (mirroring today's log_sigmoid); the GPU consumes only precomputed dlp offsets and runs the IDENTICAL logsumexp reductions. => NO new f32 softmax surface, NO new parity risk class. dlp magnitudes O(1-10) like today's logp1-logp0; low-prob categories give large-negative dlp (~-40) that exp-underflow to 0 identically in f32/f64. Parity stays ~1e-4 relative. M-step (build_tables f64, all gradients f64) and final EAP stay CPU-f64.

Reductions lp_pass (L104), nbar_pass (L145), and the online logsumexp over t/x are UNCHANGED -- cell_l returns the same scalar; nbar (person count) is category-free so nbar_pass is byte-identical.

E-step SHADER (18 -> 19 bindings):
- binding 2 (logp1) -> `dlp: array<f32>` [ctx][item][(Kmax-1)][t][x]. binding 1 (logp0) -> `lp0`, binding 3 (c0) unchanged shape/role.
- ADD binding 18 `pos_cats: array<u32>` parallel to pos_items (binding 9): category k>=1 of each non-baseline response. miss stays item-only. Uniforms gains `n_cat: u32` (reuse the _pad slot). Update the (0..18) layout loop bound (L244), keep the read/write classification (12|13|15 read_write), and the score shader 19->20.
- cell_l (L83-101, ~4 lines): v=c0[...]; pos loop k=pos_cats[j], idx=((s*n_items+i)*(n_cat-1)+(k-1))*cell + t*n_x+x, v += dlp[idx]; miss loop v -= lp0[base0]. Structure identical, K=2 identical numbers.
Note: for binary, dlp is host-computed f64(logp1)-f64(logp0) then cast to f32, vs old device-side f32(logp1)-f32(logp0); shifts GPU BINARY by ~1e-7 (f32 eps) -- well inside the existing ~1e-4 GPU-vs-CPU tolerance and NOT the bit-parity gate (that gate is CPU-only). Call it out in the GPU test.

rbar_k on GPU -- REUSE item_pass (L176-204) VERBATIM, ZERO new WGSL (all three designs converge, verified): the host bins persons by chosen category into K-1 item-major CSR lists (item_off_k / item_persons_k -- the direct generalization of today's single positive list, which is category-1-only). run_reduce is dispatched K-1 times, each binding the k-th CSR and pointing out_acc at rbar[k]'s slice. Categories are fixed across EM iterations, so binning is one-time. mbar stays ONE item_pass over the category-free miss CSR; nbar_pass unchanged. out_acc sizes [ctx][item][t][x]*(K-1). Total work ~ one sparse pass over non-baseline responses, split by category.

Host (e_step_gpu L392-589 / GpuEStepInputs L336-358): build dlp instead of logp1; add pos_cats + K-1 per-category item CSRs; loop the rbar dispatch over categories. score_pass SHADER (L631-716 / GpuScoreInputs): same cell_l edit (dlp + pos_cats + n_cat); the EAP-moment loop is category-free. Existing bounds n_dims/latent_dim<=8, q_t<=41 unchanged; add Kmax<=~16 for buffer sizing.

Ragged K_i: pad to global Kmax (unused slots filled -1e30 on host so exp->0 with no WGSL branch); add a per-item n_cat u32 binding only if memory measurably bites. JML/mmle GPU path (gpu.rs) stays binary-only (hard-error guard).

## Rust data model

Gate everything by an ORTHOGONAL response axis so the binary hot path is provably unchanged and ModelType does NOT explode into a K x interaction grid (all three designs agree; keep the `minimal-diff` enum shape + `gpu-first`'s n_cat field):

- lib.rs: `pub enum ResponseModel { Bernoulli, Gpcm, Nominal }` (Copy/Eq, default Bernoulli). Add to ModelConfig (L167): `response_model: ResponseModel`, `n_cat: usize` (=max K_i, default 2), `cat_counts: Option<Vec<usize>>` (None => uniform K). ModelType (L19) UNCHANGED -- stays the interaction/slope axis; GPCM x Distance x free_alpha = GPCM-LSIRM, GPCM x Distance x Mlsrm(fixed alpha) = PCM compose for free. model_exec_flags (L84): free_alpha=false when response_model==Nominal (pins a_i=1). Poly permitted only for the marginal estimator.

- Tables (marginal.rs L294-298): {logp1,logp0} -> { dlp: Vec<f64>, lp0: Vec<f64>, c0: Vec<f64>, n_cat: usize }. dlp = [ctx][item][(n_cat-1)][t][x] holding psi_k; binary => 1 column (dlp=eta). UNIFY (no PolyTables variant): the logsumexp/nbar/item reductions are identical for K=2 and K>2; duplicating ~200 lines of WGSL + person_pass to keep the name logp1 is copy-paste rot. The binary bit-parity test (Bernoulli fills dlp via log_sigmoid) locks the unification. Two scoring.rs readers update: eapsum_tables `tables.logp1[i*cell+c].exp()` -> read dlp[i*(K-1)*cell + 0*cell + c] as psi_1 (P_1/P_0 ratio path).

- ResponseIndex (L414-417): `pos: Vec<Vec<usize>>` + parallel `pos_cat: Vec<Vec<u8>>`; miss unchanged. index_responses (L419-433) pushes (i,cat) only when observed cat>=1.

- EStep.rbar (L523-540): stride x(K-1), idx ((s*n_items+i)*(K-1)+(k-1))*cell + t*n_x+x. nbar, mbar UNCHANGED (category-free, dimension-pooled). accumulate_person (L545-597) routes post into rbar[k-1] by observed category.

- Item params (lib.rs Params/Gradients L224-242, MarginalResult L176-209): `b` -> flattened additive intercepts n_items*(n_cat-1), row-major b[i*(K-1)+(v-1)] (K=2 => 1/item, same layout). Add `s: Vec<f64>` scoring weights n_items*(n_cat-1), populated only for Nominal (GPCM s implicit=k, stored empty). GPCM alpha stays len n_items; nominal alpha unused. Provide slope_stride(response_model)+intercept_stride(K) helpers so every site derives the stride once (guards the K-1 off-by-one that would corrupt silently across anchors/init/validate).

- ItemBank (scoring.rs L21-33): `b` reinterpreted 2-D n_items x (K-1); add n_cat, cat_counts, response_model, optional s. EAP/PV follow person_pass automatically; MAP/Lord-Wingersky/information are Phase 4.

- validate (L1605 / py L394): allow y in 0..n_cat-1. ZI all_zero (L1819): structural zero = every observed response in category 0 (pos[p].is_empty() still holds). n_free_parameters (marginal.rs L157): +(K-2) GPCM; nominal +(K-1)+(K-1)-1.

- Anchors/FIPC (fit_marginal_anchored): anchors carry K-1 thresholds/item (+K-1 scoring for nominal); route through slope_stride helper.

- PyO3 bridge (fast-mlsirm-py/src/lib.rs -- OMITTED by base designs, required for Python reach): fit_marginal (L235)->core_fit_marginal_full (L362) must plumb response_model/n_cat/cat_counts into ModelConfig (L105/276/987) and accept/return flat 2-D b + s (L344 area); parse_model_type (L1428) unchanged (response_model is a new arg, not a ModelType string).

- Python mirror: types.py MLSIRMParams.b -> 2-D thresholds + optional s; marginal.py tracks Rust line-for-line.

## Parity & tests

Parity oracle = the NumPy mirror (python/fast_mlsirm/estimators/marginal.py), already pinned line-for-line to Rust by its docstring; extend it to K FIRST, then diff Rust against it (the discipline all three designs share).

1. BINARY REGRESSION GATE (highest priority, the unification guard): response_model=Bernoulli must make the ENTIRE existing binary suite (crates/mlsirm-core/tests/marginal_recovery.rs + proptest_neg_loglik.rs + scoring/oakes/fitstats tests) pass BIT-IDENTICAL. Achieved by construction: Bernoulli fills dlp via the existing log_sigmoid (hoisted subtraction), so no arithmetic changes. Cross-check dlp[0]==eta. GPU binary rides ~1e-7 inside the existing 1e-4 tolerance (note, not gate).
2. Land the Tables{dlp,lp0,c0} refactor as a VALIDATED NO-OP first (step 2): full binary suite green BEFORE any polytomous math lands.
3. GPCM-K=2 REDUCTION TEST: fit response_model=Gpcm K=2 (softmax path) on the binary fixture; must equal the Bernoulli fit to relative ~1e-12 (resolves the byte-parity tension: softmax path exercised and shown to reduce, without the default hot path riding it). Do NOT claim bit-identical for this path (logsumexp vs log_sigmoid differ at ~1e-13).
4. NumPy golden parity (GPCM then nominal, K=3 and K=4, tiny fixture ~6 persons x 4 items): Rust CPU == Python to ~1e-9 on (a) per-node cell log-lik from build_tables/person_pass, (b) E-counts nbar/rbar_k/mbar, (c) M-step gradient g_c/g_alpha/g_zeta/g_tau/g_delta and Fisher diag. Primary correctness gate.
5. GPCM == Nominal(s frozen=k): fit nominal with scoring locked to integers; must equal the GPCM fit -- validates the scoring generalization shares one code path.
6. RECOVERY (Monte Carlo, extend marginal_recovery.rs with GPCM/nominal simulators): simulate GPCM from known (alpha, additive intercepts, zeta, tau), fixed seed, N~2000; fit; RMSE < ~0.1 per param family, theta-EAP correlation > ~0.95. Repeat nominal (scoring/intercept recovery up to the a_i=1 identification).
7. GPU-vs-CPU: rbar_k, lp within ~1e-4 relative at K=3; verify the K-1 dispatch rbar sums equal CPU rbar_k.
8. HAND CHECK (ponytail self-check, one runnable assert per nontrivial branch): a 3-category softmax with known psi -> assert sum P_k=1, resid sum-to-zero, V>=0, and R by hand; assert K=2 GPCM residual == r-n*prob.
9. ORDERED-vs-UNORDERED gate: assert lord_wingersky / eapsum REJECT response_model=Nominal (summed score not sufficient for unordered categories) -- else summed-score EAP is silently wrong.
10. Edge cases: empty category (r_k=0 -> intercept driven by -n*P_k, clamp like alpha [-6,3], floor P_k, assert no NaN); ragged K_i (-1e30 sentinel path); all-responses-in-one-category item; missing interleaved with polytomous; ZI redefinition; anchored polytomous item; K=2 nominal == binary.
11. External cross-check (loose): mirt R GPCM on a public Likert set; item params within a few %. (No external oracle exists for the k-scaled latent-space term -- recovery sim is the only check there.)
12. SE/fit (Phase 4): oakes.rs SEs finite and sane on the GPCM recovery fit; fitstats item-fit correct for a known-fit GPCM item.

## Files touched

- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/lib.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/marginal.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/gpu_marginal.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/scoring.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/oakes.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/fitstats.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/gpu.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/src/mmle.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/mlsirm-core/tests/marginal_recovery.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/crates/fast-mlsirm-py/src/lib.rs
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/python/fast_mlsirm/estimators/marginal.py
- C:/Users/Seongho Bae/lsirt/fast-mlsirm/python/fast_mlsirm/estimators/types.py

## Risks

- MODELING (needs maintainer sign-off before build): the latent-space term is forced to enter psi_k scaled by s_k (a category-constant space term has zero gradient because sum_k resid_k=0), making the LSIRM space axis category-monotone. Polytomous-LSIRM is novel; no external oracle -- correctness is guaranteed only vs the NumPy mirror + recovery sim, and the GPCM/nominal cores match Muraki/Bock.
- PARITY HAZARD (verified, do NOT introduce): keep the existing `if n<=0.0 && r<=0.0 continue` guard (marginal.rs L1050/1138/1306/1406, oakes.rs L164); `minimal-diff`'s `if n<=0 continue` drops n~0,r>0 nodes and perturbs even the binary trajectory. Generalize to `n<=0 && all r_k<=0`.
- COVARIATE bug for K>2: the ItemCovariate/delta offset must move from flat eta (build_tables L388, m_step_delta) into `base`, or it cancels in the softmax and silently nulls the covariate. Fix folds it into base with g_delta=R*w.
- SCOPE: this is a 5-phase PR series, not one PR. The default-Bernoulli flag lets each phase land green independently, but bundling GPU+nominal+scoring+SE+Python into one review is unrealistic. Phase 4 (scoring/SE) is NOT optional for an ordered-rubric deliverable -- Phase 1 fits GPCM but cannot score/report.
- OMITTED-FILE debt (now in files_touched): oakes.rs (SEs: per_item L49-52 no thresholds, resid=r-n*sigmoid L188, y=1.0 cross-term ~L423) and fitstats.rs (item fit, per_item L1299) independently re-implement the binary path and need the K-1 threshold/scoring params, the category/score-weighted residual, and polytomous rbar. SEs are first-class output; neither is auto-derived from person_pass.
- SOFTMAX overflow at grid edges: the host K-way softmax MUST max-subtract or it infs at extreme |theta|*s_k; log_sigmoid was inherently stable.
- NOMINAL identification: freeing both a_i and s_k is non-identified -> pin a_i=1 (free_alpha=false); label-switching mitigated by baseline-category constraint + lambda penalties + ordered init s_k=k. Document.
- rbar memory grows x(K-1) (c0/nbar/mbar unchanged, so total grows sublinearly); fine for Likert K<=7, note for large item banks x multilevel contexts. M-step cost ~Kx per node with backtrack re-eval -- acceptable but hot.
- JML/mmle path stays binary-only: add a HARD error/return (NOT a debug_assert like assert_distance_kind, a no-op in release) in lib.rs neg_loglik_and_grad, mmle.rs, gpu.rs rejecting response_model!=Bernoulli. Polytomous is MMLE/marginal-only.
- Diagonal-only intercept preconditioner drops -n*P_m*P_l cross-terms; may need more inner M-steps at large K. Armijo on true q backstops; upgrade to a (K-1)x(K-1) Newton block only if it stalls.
- Phase-4 scoring is genuinely new math, not a gate: polytomous Lord-Wingersky must convolve a 0..K-1 per-item category distribution (score range 0..sum(K_i-1)), gated to ordered GPCM; eapsum table sizes grow; score_map needs a GPCM Newton gradient/Hessian.
- Variable K-1 stride ripples into anchors/FIPC, init b, validate, n_free_parameters, ItemBank; route every site through slope_stride/intercept_stride helpers or a silent off-by-one corrupts params.

## Implementation steps

- PHASE 1 -- CPU estimator core (shippable: fits GPCM on CPU). Step 1: lib.rs add ResponseModel{Bernoulli,Gpcm,Nominal} + ModelConfig fields (response_model,n_cat,cat_counts), default Bernoulli/K=2 so all existing code compiles and the binary suite stays green; model_exec_flags pins free_alpha=false for Nominal; add slope_stride/intercept_stride helpers; add HARD-error guards in neg_loglik_and_grad/mmle.rs/gpu.rs rejecting non-Bernoulli; relax validate L1605 + py L394 to 0..K-1.
- Step 2: refactor Tables{logp1,logp0}->{dlp,lp0,c0,n_cat}; build_tables_offset branches on response_model (Bernoulli fills dlp via log_sigmoid = hoisted subtraction, bit-identical); update the two scoring.rs readers. RUN FULL BINARY SUITE -> must be bit-green (proves the refactor is a no-op).
- Step 3: build_tables_offset Gpcm/Nominal branch = host f64 max-subtract K-way softmax (psi via eta_at_kind with b=0); fold the covariate offset into base. index_responses/ResponseIndex category-tagged pos + pos_cat; person_pass L453-467 `+= dlp[k-1]` decomposition (miss + logsumexp unchanged). GPCM-K=2 reduces to Bernoulli within 1e-12; K=3 vs NumPy to 1e-9.
- Step 4: EStep.rbar -> rbar_k (x(K-1)); accumulate_person routes post by observed category; nbar/mbar untouched. item_q multinomial objective; m_step_items resid_k, score-weighted R, K-1 intercept grads g_{c,m}=resid_m, V=n*Var_P(s) diagonal Fisher, d_b vector, keep guard + Armijo + alpha clamp. m_step_tau/m_step_delta use R,V. init additive intercepts from cumulative category proportions. n_free_parameters bump. GPCM recovery test (extend marginal_recovery.rs with a GPCM simulator).
- Step 5: extend the NumPy mirror (marginal.py, types.py) in lockstep -- softmax/dlp/rbar_k/gradient/init/validation; run the Rust<->Python golden harness (GPCM K=3,4) to 1e-9. Closes Phase 1.
- PHASE 2 -- GPU E-step. Step 6: Uniforms.n_cat; swap logp1->dlp + add pos_cats binding (18->19 layout, score 19->20); edit cell_l in both shaders (~4 lines); build K-1 per-category item CSRs on the host and dispatch item_pass K-1 times into rbar[k] (zero new WGSL); score_pass cell_l edit. GPU-vs-CPU parity K=3 to ~1e-4; note the ~1e-7 binary GPU shift.
- PHASE 3 -- Nominal. Step 7: a_i=1 via model_exec_flags; free scoring s_k with g_{s,m}=resid_m*base and R=sum s_k*resid_k; store s in Params/ItemBank/py; identification via baseline + prior + ordered init. Tests: Nominal(s=k)==GPCM, nominal recovery.
- PHASE 4 -- Scoring + SE (NOT optional for a polytomous scoring deliverable). Step 8: scoring.rs polytomous lord_wingersky (per-item 0..K-1 category convolution, score range 0..sum(K_i-1)), gated to ordered GPCM (reject Nominal); eapsum_tables expanded score range + category probs; score_map GPCM/nominal Newton (score-weighted residual, Hessian -a^2*Var_P(s)); item_information a^2*Var_P(s); plausible_values/score_eap follow person_pass automatically. Step 9: oakes.rs SEs -- extend ParamVec.per_item to K-1 thresholds (+scoring), the category/score-weighted Q-gradient, and polytomous rbar cross-terms; fitstats.rs polytomous item-fit. SE finiteness + fit-stat tests on the recovery fit.
- PHASE 5 -- Python reach. Step 10: PyO3 bridge fast-mlsirm-py/src/lib.rs -- plumb response_model/n_cat/cat_counts through fit_marginal->core_fit_marginal_full->ModelConfig and flat 2-D b + s in/out; estimator API + types.py + docs. Ship GPCM (covers PCM via fixed slope) as the built default; nominal shares the same softmax cell + a ~20-line distinct gradient block.

---

## Literature resolution of the space-scaling design fork (2026-07-14)

The synthesized spec flagged that under an adjacent-category **softmax (GPCM /
nominal)** cell a category-constant term cancels, so identification *forces* the
latent-space interaction `I(x) = -gamma*d(z,w)` to enter category-scaled
(`s_k * I(x)`), making the latent-space axis category-ordered. A literature
search (alphaXiv + Consensus) resolves this:

- **Jeon, Jin, Schweinberger & Baugh (2021)**, *Mapping unobserved
  item-respondent interactions: A latent space item response model*
  (Psychometrika; arXiv:2007.08719) — the base LSIRM adds the interaction as a
  **single scalar** on the linear predictor:
  `logit P(Y=1) = alpha_j + beta_i - gamma*d(z_k, w_i)`. The paper states the
  polytomous extension is "straightforward, by replacing the logit-link ... by a
  suitable link function ..., as in generalized linear models" — i.e. the GLM /
  link route, NOT a bespoke softmax.
- **Go, Kim, Park, Park, Jeon & Jin (2024/2025)**, *lsirm12pl: An R package for
  the latent space item response model* (R Journal; arXiv:2205.06989) — the
  authors' own package — extends LSIRM to **continuous** responses with an
  identity link: `y = theta + beta - gamma*d(z,w) + eps`; again a single additive
  interaction, now on the mean. They explicitly list ordinal LSIRM as
  **in-progress future work** ("we are currently engaged in the development of ...
  ordinal and longitudinal data"). So no published ordinal/polytomous LSIRM
  exists yet: this is novel territory, and the original authors' intended route is
  a link function, not a softmax.

**Resolution.** For the LSIRM family the identification-clean polytomous
extension is the **cumulative-logit Graded Response Model (Samejima 1968)**, not
the adjacent-category GPCM softmax:

    logit P(Y >= k | theta, x) = a_i*theta + beta_{i,k} - gamma*d(z_k, w_i),  k = 1..K-1
    P(Y = k) = P(Y >= k) - P(Y >= k+1)

The single interaction `-gamma*d` enters every cumulative logit as a **shared
shift**. Cumulative-logit is NOT translation-invariant, so nothing cancels and NO
category-scaling is forced; the space axis keeps its original person-item
interaction-map meaning, exactly matching the binary (`-gamma*d` on the logit) and
continuous (`-gamma*d` on the mean) cases. This is the model the original authors
point to.

- **GPCM** (adjacent-category softmax; Muraki 1992) remains a valid alternative
  but carries the forced `s_k * I(x)` category-scaling — keep it as a documented
  option for partial-credit scoring where score-weighting is intended, not the
  default.
- **Nominal** (Bock 1972; multidimensional: Revuelta 2014; Falk & Cai 2015)
  genuinely uses category-specific scoring functions, so category-specific space
  entry is consistent with that model's philosophy.

**Implementation implication.** Target **GRM-LSIRM (cumulative-logit)** as the
default polytomous model. The GRM cell replaces the softmax cell:
`P(Y=k) = Phi_k - Phi_{k+1}` with `Phi_k = sigmoid(a*theta + beta_{i,k} - gamma*d)`;
the softmax `category_logprobs`/`gpcm_node_gradient` oracle is retained for the
GPCM/nominal options only. Which model is primary depends on the target items' response
format: ordinal Likert / rubric levels -> GRM; partial-credit performance levels
-> GPCM.
