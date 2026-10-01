//! Unit contract for the single-group polytomous two-tier GRM fitter
//! (`mlsirm_core::two_tier_grm`, stage 4 of #1912).
//!
//! * Dimension-reduction exactness: the reduced marginal log-likelihood must
//!   equal brute-force full product-grid integration to `<= 1e-10` on a tiny
//!   two-primary model (10 items, `P = 2`, `S = 2`, `Q = 7`). The two paths
//!   reorder the same finite sum, so any larger gap is an implementation bug,
//!   not quadrature error.
//! * Reduction to stage 1: at `P = 1` (every item loading the single
//!   primary) the reduced marginal log-likelihood must equal the stage-1
//!   `bifactor_grm_marginal_loglik` to `<= 1e-10` on the same parameters.
//! * Argument validation and failure reporting: unsupported quadrature,
//!   bad iteration/tolerance/start budgets, malformed primary/specific maps,
//!   under-loaded primaries, out-of-range categories, unobserved categories,
//!   non-PD `phi` in the oracle, and the non-convergence report
//!   (`converged == false`, `termination_reason == "max_iter_reached"`).
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (the posterior-moment test below
//! uses directly inspected pp. 608-609, Appendices A/B)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use crate::two_tier_grm::{
    fit_two_tier_grm, fit_two_tier_grm_with_device, two_tier_grm_marginal_loglik,
    two_tier_grm_marginal_loglik_brute, TwoTierGrmConfig,
};
use crate::Device;

// 테스트 전용 thread-local 궤적: production 빌드에는 포함하지 않는다.
struct ReferenceTracePoint {
    iteration: usize,
    params: Vec<crate::two_tier_grm::ItemParams>,
    item13_counts: Vec<Vec<f64>>,
    loglik: f64,
}
thread_local! {
    static REFERENCE_TRACE: std::cell::RefCell<Option<Vec<ReferenceTracePoint>>> = const {
        std::cell::RefCell::new(None)
    };
}
#[cfg(all(feature = "gpu", not(coverage)))]
thread_local! {
    static REFERENCE_GPU_TIMINGS: std::cell::RefCell<Option<Vec<(&'static str, f64)>>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn reference_gpu_timing_enabled() -> bool {
    REFERENCE_GPU_TIMINGS.with(|timings| timings.borrow().is_some())
}

#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) fn record_reference_gpu_timing(name: &'static str, seconds: f64) {
    REFERENCE_GPU_TIMINGS.with(|timings| {
        if let Some(rows) = timings.borrow_mut().as_mut() {
            rows.push((name, seconds));
        }
    });
}

thread_local! {
    static TABLE_GRM_CALLS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

pub(crate) fn record_table_grm_call() {
    TABLE_GRM_CALLS.with(|calls| {
        if let Some(n) = calls.get() {
            calls.set(Some(n + 1));
        }
    });
}

/// 기존 scalar 범주 평가를 oracle로 유지한다. Cai (2010, pp. 608–609,
/// Appendices A/B)의 동일 node 확률을 범주 순서대로 보존하는 검사다.
/// 참고문헌: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. *Psychometrika, 75*(4), 581–612.
/// https://doi.org/10.1007/s11336-010-9178-0.
#[test]
fn item_tables_preserve_scalar_bits_with_one_grm_call_per_node() {
    use super::{item_cat_logprob, item_logprob_tables, ItemParams, Validated};
    let coords = [1.0, 0.0, -2.0, 0.7, 3.1, -1.3];
    let ts = [-1.7, 0.0, 0.6];
    let mut cases = 0;
    let mut zero_mass_cells = 0;
    for specifics in [false, true] {
        for thresholds in [vec![0.3], vec![1.2, 0.0, -1.2], vec![1e-18, 0.0]] {
            let v = Validated {
                n_persons: 1,
                n_items: 4,
                n_primary: 2,
                n_specific: usize::from(specifics),
                n_cat: thresholds.len() + 1,
                m1: thresholds.len(),
                grid_size: 3,
                free_primaries: vec![vec![0, 1], vec![1, 0], vec![0], vec![1]],
                blocks: if specifics { vec![vec![0, 1]] } else { vec![] },
                specific_free: if specifics { vec![2, 3] } else { vec![0, 1, 2, 3] },
                item_block: if specifics {
                    vec![Some(0), Some(0), None, None]
                } else {
                    vec![None; 4]
                },
            };
            let mut params: Vec<ItemParams> = [
                (vec![0.9, -0.3], -0.7),
                (vec![-1.2, 0.7], 0.8),
                (vec![0.5, 0.0], 0.0),
                (vec![0.0, -0.8], 0.0),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, (a_p, a_s))| ItemParams {
                a_p,
                a_s: v.item_block[i].map(|_| a_s),
                d: thresholds.clone(),
            })
            .collect();
            for fresh in [false, true] {
                if fresh {
                    for par in &mut params {
                        for a in &mut par.a_p { *a *= -1.13; }
                    }
                }
                TABLE_GRM_CALLS.with(|calls| calls.set(Some(0)));
                let result = item_logprob_tables(&v, &params, &coords, &ts, 3);
                let actual_calls = TABLE_GRM_CALLS.with(|calls| calls.replace(None).unwrap());
                let tables = result.unwrap();
                let nodes: usize = v.item_block.iter()
                    .map(|block| 3 * if block.is_some() { ts.len() } else { 1 })
                    .sum();
                println!("table_helper_calls={actual_calls}, nodes={nodes}, n_cat={}", v.n_cat);
                assert_eq!(actual_calls, nodes, "전체 범주 확률의 node별 중복 계산");
                for i in 0..v.n_items {
                    let hs = if v.item_block[i].is_some() { ts.len() } else { 1 };
                    assert_eq!(tables[i].len(), 3 * hs * v.n_cat);
                    for g in 0..3 {
                        for h in 0..hs {
                            for cat in 0..v.n_cat {
                                let expected = item_cat_logprob(&v, &params, &coords, &ts, i, g, h, cat);
                                let actual = tables[i][(g * hs + h) * v.n_cat + cat];
                                assert_eq!(actual.to_bits(), expected.to_bits());
                                zero_mass_cells += usize::from(actual == f64::NEG_INFINITY);
                            }
                        }
                    }
                }
                cases += 1;
            }
            TABLE_GRM_CALLS.with(|calls| calls.set(Some(0)));
            assert!(item_logprob_tables(&v, &params, &coords, &ts, usize::MAX).is_err());
            assert_eq!(TABLE_GRM_CALLS.with(|calls| calls.replace(None).unwrap()), 0);
        }
    }
    assert_eq!(cases, 12);
    assert!(zero_mass_cells > 0);
    println!("table_scalar_bit_cases={cases}, legal_zero_mass_cells={zero_mass_cells}, overflow_rejected_before_probability=true");
}

pub(super) fn record_reference_trace(
    iteration: usize,
    params: &[crate::two_tier_grm::ItemParams],
    counts: &[Vec<Vec<f64>>],
    loglik: f64,
) {
    REFERENCE_TRACE.with(|trace| {
        if let Some(rows) = trace.borrow_mut().as_mut() {
            rows.push(ReferenceTracePoint {
                iteration,
                params: params.to_vec(),
                item13_counts: counts[13].clone(),
                loglik,
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Shared tiny two-tier problem: 10 items, P = 2 primaries in simple
// structure (items 0-4 on primary 0, items 5-9 on primary 1), S = 2
// specifics cross-cutting the primary split (method-factor layout).
// ---------------------------------------------------------------------------

const TINY_N_ITEMS: usize = 10;
const TINY_N_PRIMARY: usize = 2;
const TINY_N_SPECIFIC: usize = 2;
const TINY_N_CAT: usize = 3;
/// Row-major `n_items x n_primary` free-slope pattern.
const TINY_PRIMARY_MAP: [bool; TINY_N_ITEMS * TINY_N_PRIMARY] = [
    true, false, // 0
    true, false, // 1
    true, false, // 2
    true, false, // 3
    true, false, // 4
    false, true, // 5
    false, true, // 6
    false, true, // 7
    false, true, // 8
    false, true, // 9
];
/// Specific blocks cross the primary split: S0 = {0, 1, 5, 6}, S1 = {2, 3, 4, 7, 8, 9}.
const TINY_SPECIFIC_MAP: [i32; TINY_N_ITEMS] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

fn tiny_params() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Row-major n_items x n_primary; exact 0.0 at fixed positions.
    let a_primary = vec![
        1.2, 0.0, //
        -0.9, 0.0, //
        1.0, 0.0, //
        0.7, 0.0, //
        1.1, 0.0, //
        0.0, 1.0, //
        0.0, 0.8, //
        0.0, 1.2, //
        0.0, -0.7, //
        0.0, 0.9,
    ];
    let a_specific = vec![0.8, 1.1, 0.9, 0.6, 0.7, 1.0, 0.8, 0.9, 0.7, 1.1];
    // Strictly decreasing within each item (n_cat - 1 = 2 intercepts).
    let thresholds = vec![
        0.9, -0.7, //
        0.5, -1.0, //
        1.1, -0.4, //
        0.2, -1.3, //
        0.8, -0.6, //
        1.0, -0.5, //
        0.6, -0.9, //
        1.2, -0.3, //
        0.4, -1.1, //
        0.7, -0.8,
    ];
    let phi = vec![1.0, 0.35, 0.35, 1.0];
    (a_primary, a_specific, thresholds, phi)
}

fn tiny_data() -> (Vec<usize>, usize) {
    // Deterministic toy responses covering every category of every item.
    let n_persons = 12usize;
    let rows: [[usize; TINY_N_ITEMS]; 12] = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        [1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        [2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
        [1, 0, 2, 1, 0, 2, 1, 0, 2, 1],
        [0, 2, 0, 1, 0, 2, 0, 1, 2, 1],
        [1, 2, 1, 0, 1, 2, 1, 0, 1, 2],
        [2, 0, 2, 1, 2, 0, 2, 1, 0, 0],
    ];
    let mut y = Vec::with_capacity(n_persons * TINY_N_ITEMS);
    for row in rows {
        y.extend_from_slice(&row);
    }
    (y, n_persons)
}

// 중간 범위의 직접 cumulative-logistic oracle. production helper를 호출하지 않는다.
#[cfg(all(feature = "gpu", not(coverage)))]
fn fixed_item_direct_objective(params: &[f64], free: &[usize], specific: bool,
    coords: &[f64], ts: &[f64], p: usize, qs: usize, counts: &[Vec<f64>]) -> f64 {
    let off = free.len() + usize::from(specific);
    let beta = &params[off..];
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut f = 0.0;
    for (node, row) in counts.iter().enumerate() {
        let (g, h) = if specific { (node / qs, node % qs) } else { (node, 0) };
        let mut base = 0.0;
        for (t, &dim) in free.iter().enumerate() { base += params[t] * coords[g * p + dim]; }
        if specific { base += params[free.len()] * ts[h]; }
        for (cat, &count) in row.iter().enumerate() {
            if count == 0.0 { continue; }
            let prob = if cat == 0 { 1.0 - sigmoid(base + beta[0]) }
                else if cat == beta.len() { sigmoid(base + beta[cat - 1]) }
                else { sigmoid(base + beta[cat - 1]) - sigmoid(base + beta[cat]) };
            f -= count * prob.ln();
        }
    }
    f
}

#[cfg(all(feature = "gpu", not(coverage)))]
fn fixed_item_return_contract(case: usize, v: &super::Validated, params: &[super::ItemParams],
    coords: &[f64], ts: &[f64], cpu: &[Vec<Vec<f64>>], gpu: &[Vec<Vec<f64>>], input_hash: &str) {
    use super::{item_neg_ll_grad, m_step_item};
    use serde_json::json;
    use sha2::Digest;
    let items: &[usize] = match case { 0 => &[0, 4], 1 => &[0], _ => &[0] };
    for &i in items {
        let free = &v.free_primaries[i];
        let specific = v.item_block[i].is_some();
        let mut packed: Vec<f64> = free.iter().map(|&d| params[i].a_p[d]).collect();
        if let Some(a) = params[i].a_s { packed.push(a); }
        packed.extend_from_slice(&params[i].d);
        let eval = |point: &[f64], counts: &[Vec<f64>]| item_neg_ll_grad(point, free,
            specific, coords, ts, v.n_primary, v.grid_size, ts.len(), counts, v.n_cat);
        let (f0, gradient) = eval(&packed, &cpu[i]);
        let (gpu_f, gpu_gradient) = eval(&packed, &gpu[i]);
        assert_eq!(f0.to_bits(), gpu_f.to_bits());
        assert_eq!(gradient, gpu_gradient);
        let direct = fixed_item_direct_objective(&packed, free, specific, coords, ts,
            v.n_primary, ts.len(), &cpu[i]);
        assert!(f0.is_finite() && direct.is_finite());
        assert!((f0 - direct).abs() < 1e-4);
        let mut fd_residual = 0.0_f64;
        for col in 0..packed.len() {
            let mut plus = packed.clone(); plus[col] += 1e-6;
            let mut minus = packed.clone(); minus[col] -= 1e-6;
            let fd = (fixed_item_direct_objective(&plus, free, specific, coords, ts,
                v.n_primary, ts.len(), &cpu[i]) - fixed_item_direct_objective(&minus,
                free, specific, coords, ts, v.n_primary, ts.len(), &cpu[i])) / 2e-6;
            fd_residual = fd_residual.max((fd - gradient[col]).abs());
        }
        assert!(fd_residual.is_finite() && fd_residual < 1e-4);
        let np = packed.len();
        let mut hessian = vec![vec![0.0; np]; np];
        for col in 0..np {
            let mut point = packed.clone(); point[col] += 1e-5;
            let g = eval(&point, &cpu[i]).1;
            for row in 0..np { hessian[row][col] = (g[row] - gradient[row]) / 1e-5; }
        }
        let raw_hessian = hessian.clone();
        for row in 0..np {
            for col in 0..np { hessian[row][col] = 0.5 * (hessian[row][col] + hessian[col][row]); }
            hessian[row][row] += 1e-8;
        }
        let cpu_return = m_step_item(packed.clone(), free, specific, coords, ts,
            v.n_primary, v.grid_size, ts.len(), &cpu[i], v.n_cat, 1e-8, 1);
        let gpu_return = m_step_item(packed.clone(), free, specific, coords, ts,
            v.n_primary, v.grid_size, ts.len(), &gpu[i], v.n_cat, 1e-8, 1);
        for (a, b) in cpu_return.iter().zip(&gpu_return) { assert_eq!(a.to_bits(), b.to_bits()); }
        let returned_f = eval(&cpu_return, &cpu[i]).0;
        let changed = cpu_return.iter().zip(&packed).any(|(a,b)| a.to_bits()!=b.to_bits());
        assert!(cpu_return.iter().all(|x| x.is_finite()) && returned_f.is_finite());
        assert!(returned_f <= f0);
        let input = json!({"counts_input_sha256":input_hash,"case":case,"item":i,
            "packed":packed,"free_primary_dimensions":free,"specific":specific,
            "counts":cpu[i],"ridge":1e-8,"newton_iter":1});
        let hash = format!("{:x}",sha2::Sha256::digest(serde_json::to_vec(&input).unwrap()));
        println!("{}",json!({"event":"item_return_case","input":input,"input_sha256":hash,
            "native_objective":f0,"direct_objective":direct,"objective_abs_delta":(f0-direct).abs(),
            "gradient":gradient,"central_fd_max_abs_residual":fd_residual,
            "raw_forward_fd_hessian":raw_hessian,"current_inplace_ridged_matrix":hessian,
            "returned_params":cpu_return,"returned_objective":returned_f,"changed":changed,
            "cpu_word_return_bits_equal":true,"branch_observed":false,
            "backtracking_branch":"not instrumented; do not infer acceptance branch",
            "fit_executed":false,"convergence":"not applicable"}));
    }
    if case == 2 {
        let free = [0usize];
        let packed = [1.0, 1e-18, 0.0];
        let zero = vec![vec![0.0; 3]];
        let (native_f, native_g) = item_neg_ll_grad(&packed,&free,false,&[1.0],&[0.0],1,1,1,&zero,3);
        let direct_f = fixed_item_direct_objective(&packed,&free,false,&[1.0],&[0.0],1,1,&zero);
        let returned = m_step_item(packed.to_vec(),&free,false,&[1.0],&[0.0],1,1,1,&zero,3,1e-8,1);
        assert_eq!(direct_f,0.0);
        assert!(native_g.iter().all(|x| x.is_finite() && *x==0.0));
        assert_eq!(returned,packed);
        for (name, row) in [("zero_weights",vec![0.0,0.0,0.0]),
            ("mixed_supported_weight",vec![1.0,0.0,0.0]),
            ("positive_impossible_weight",vec![0.0,1.0,0.0])] {
            let counts = vec![row];
            let (f,g) = item_neg_ll_grad(&packed,&free,false,&[1.0],&[0.0],1,1,1,&counts,3);
            let independent = fixed_item_direct_objective(&packed,&free,false,&[1.0],&[0.0],1,1,&counts);
            let ret = m_step_item(packed.to_vec(),&free,false,&[1.0],&[0.0],1,1,1,&counts,3,1e-8,1);
            if name == "positive_impossible_weight" {
                assert!(f.is_infinite() && f.is_sign_positive());
                assert!(independent.is_infinite() && independent.is_sign_positive());
            } else { assert!(f.is_nan() && independent.is_finite()); }
            assert_eq!(ret,packed);
            println!("{}",json!({"event":"zero_weight_boundary_case","case":name,
                "packed":packed,"coords":[1.0],"counts":counts,
                "native_objective_is_nan":f.is_nan(),"native_objective_positive_inf":f==f64::INFINITY,
                "independent_objective_finite":independent.is_finite(),
                "independent_objective":if independent.is_finite(){Some(independent)}else{None},
                "gradient_finite":g.iter().all(|x|x.is_finite()),"returned_unchanged":true,
                "counts_domain":"finite nonnegative; malformed private counts not executed",
                "fit_executed":false,"convergence":"not applicable"}));
        }
        println!("{}",json!({"event":"zero_count_legal_zero_probability_observation",
            "packed":packed,"coords":[1.0],"counts":zero,"native_objective_finite":native_f.is_finite(),
            "native_objective_is_nan":native_f.is_nan(),"direct_zero_count_objective":direct_f,
            "native_gradient":native_g,"returned_unchanged":true,"fit_executed":false,
            "convergence":"not applicable","production_modified":false}));
    }
}

// 후보의 log probability 경계를 CPU만으로 검사한다. 제안 규칙은 runtime에 연결하지 않는다.
#[cfg(all(feature = "gpu", not(coverage)))]
fn fixed_item_candidate_contract() {
    use super::{item_neg_ll_grad, m_step_item};
    use crate::poly::grm_logprobs;
    use serde_json::json;
    use sha2::Digest;
    let proposed_term = |count: f64, lp: f64| {
        if count == 0.0 && lp == f64::NEG_INFINITY { 0.0 } else { count * lp }
    };
    let free = [0];
    for (name, packed, row, valid) in [
        ("legal_zero_weights", vec![1.0,1e-18,0.0], vec![0.0,0.0,0.0], true),
        ("legal_mixed_supported", vec![1.0,1e-18,0.0], vec![1.0,0.0,0.0], true),
        ("positive_impossible", vec![1.0,1e-18,0.0], vec![0.0,1.0,0.0], true),
        ("unordered_nan_zero_weight", vec![1.0,-0.2,0.2], vec![1.0,0.0,0.0], false),
    ] {
        let counts = vec![row];
        let lp = grm_logprobs(packed[0],&packed[1..]);
        let (native,gradient) = item_neg_ll_grad(&packed,&free,false,&[1.0],&[0.0],1,1,1,&counts,3);
        let proposed = -counts[0].iter().zip(&lp).map(|(&r,&l)| proposed_term(r,l)).sum::<f64>();
        let returned = m_step_item(packed.clone(),&free,false,&[1.0],&[0.0],1,1,1,&counts,3,1e-8,1);
        for (a,b) in returned.iter().zip(&packed) { assert_eq!(a.to_bits(),b.to_bits()); }
        let direct = if valid { Some(fixed_item_direct_objective(&packed,&free,false,
            &[1.0],&[0.0],1,1,&counts)) } else { None };
        if name.starts_with("legal_") {
            assert!(native.is_nan() && proposed.is_finite());
            assert!((proposed-direct.unwrap()).abs() < 1e-12);
        } else if name == "positive_impossible" {
            assert_eq!(native,f64::INFINITY);
            assert_eq!(proposed,f64::INFINITY);
        } else {
            assert!(lp[1].is_nan() && counts[0][1]==0.0);
            assert!(native.is_nan() && proposed.is_nan());
        }
        let input = json!({"case":name,"packed":packed,"counts":counts,"coords":[1.0],
            "free":[0],"n_cat":3,"valid_model_state":valid,
            "invalid_state_role":if valid {"not applicable"}else{"unordered line-search candidate; not admitted model"}});
        println!("{}",json!({"event":"candidate_rule_case","input_sha256":format!("{:x}",
            sha2::Sha256::digest(serde_json::to_vec(&input).unwrap())),"input":input,
            "log_probability_bits":lp.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
            "native_objective_finite":native.is_finite(),"native_objective_nan":native.is_nan(),
            "native_objective_positive_inf":native==f64::INFINITY,
            "native_gradient_bits":gradient.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
            "native_gradient_finite":gradient.iter().all(|x|x.is_finite()),"native_return_unchanged":true,
            "proposed_objective_finite":proposed.is_finite(),"proposed_objective_nan":proposed.is_nan(),
            "proposed_objective_positive_inf":proposed==f64::INFINITY,
            "proposed_objective":if proposed.is_finite(){Some(proposed)}else{None},
            "independent_objective":direct.filter(|x|x.is_finite()),
            "rule":"count==0 AND lp==-inf only; other products and sum order unchanged",
            "rule_runtime_connected":false,"gpu_context_initialized":false,"fit_executed":false,
            "internal_branch":"not instrumented","convergence":"not applicable"}));
    }
    // +inf는 합법 GRM log probability가 아니다. 인위적 term 입력을 native 모형으로 표시하지 않는다.
    for (name, count, lp) in [("zero_positive_inf",0.0,f64::INFINITY),
        ("zero_nan",0.0,f64::NAN),("positive_negative_inf",1.0,f64::NEG_INFINITY)] {
        let native_product = count * lp;
        let proposed_product = proposed_term(count,lp);
        assert!(!native_product.is_finite() && !proposed_product.is_finite());
        if count==0.0 { assert!(native_product.is_nan() && proposed_product.is_nan()); }
        println!("{}",json!({"event":"synthetic_nonfinite_term_case","case":name,
            "count_bits":count.to_bits(),"log_probability_bits":lp.to_bits(),
            "native_product_bits":native_product.to_bits(),"proposed_product_bits":proposed_product.to_bits(),
            "native_item_objective_evaluated":false,"native_item_gradient_evaluated":false,
            "native_item_return_evaluated":false,"domain":"synthetic scalar term; not valid GRM log probability",
            "nonfinite_preserved":true,"production_modified":false,"fit_executed":false}));
    }
    println!("candidate_rule_native_cases=4, synthetic_nonfinite_terms=3, runtime_connected=false, gpu=false, fit=false");
}

// 기존 Cai (2010, pp. 608–609) oracle를 재사용하는 고정 모수 검사다.
// 기존 owned profile의 명시적 로컬 모드이며 fit·M-step은 실행하지 않는다.
#[cfg(all(feature = "gpu", not(coverage)))]
fn reference_gpu_fixed_bank_counts_contract(qp: usize, qs: usize, gpu_budget: u64, host_budget: u64, item_update: bool) {
    use super::{build_primary_grid, e_step_gpu_person_moments, e_step_with_moments,
        gh_rule, item_logprob_tables, validate_data, ItemParams, ReducedFitStatistics};
    use serde_json::json;
    use sha2::Digest;
    // 가장 큰 case의 caller 예산을 GH/table/counts 할당보다 먼저 확인한다.
    let g = qp.checked_mul(qp).expect("counts grid overflows");
    let nodes = 16usize.checked_mul(g).and_then(|v| v.checked_mul(qs)).expect("counts nodes overflow");
    let payload = nodes.checked_mul(4).and_then(|v| v.checked_mul(8)).expect("counts bytes overflow");
    let headers = nodes.checked_add(17).and_then(|v| v.checked_mul(std::mem::size_of::<Vec<f64>>())).expect("counts headers overflow");
    let output = 4usize.checked_mul(g).and_then(|v| v.checked_mul(qs)).and_then(|v| v.checked_add(g)).expect("counts output overflows");
    let preflight = payload.checked_add(headers).and_then(|v| v.checked_mul(6))
        .and_then(|v| payload.checked_mul(3).and_then(|t| v.checked_add(t)))
        .and_then(|v| output.checked_mul(256).and_then(|t| v.checked_add(t)))
        .and_then(|v| v.checked_add(65536)).expect("counts host estimate overflows");
    assert!(preflight as u64 <= host_budget, "counts contract host budget is insufficient");
    let ctx = crate::gpu::GpuContext::get().expect("hardware GPU required for counts contract");
    assert!(matches!(ctx.adapter_info.device_type, wgpu::DeviceType::DiscreteGpu
        | wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu));
    let cfg = TwoTierGrmConfig { q_primary: qp, q_specific: qs,
        estimate_primary_correlation: false, ..valid_config() };
    let (tz, wz) = gh_rule(qp).unwrap();
    let (ts, ws) = gh_rule(qs).unwrap();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let mut positive_cases = 0;
    let mut rejected_cases = 0;
    for case in 0..3 {
        let (p, s, j, k, n) = (if case == 2 { 1 } else { 2 }, if case == 2 { 0 } else { 4 }, 16, 4, 4);
        let row = [0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2];
        let mut y: Vec<usize> = (0..n).flat_map(|person| row.iter().map(move |&v| (v + person) % k)).collect();
        let wording = [4, 5, 6, 9, 12, 14, 15];
        let pm: Vec<bool> = (0..j).flat_map(|i| (0..p).map(move |d| d == 0 || wording.contains(&i))).collect();
        let sm: Vec<i32> = (0..j).map(|i| if s == 0 || (case == 1 && (i == 0 || i == 15)) { -1 } else { (i / 4) as i32 }).collect();
        let mut params: Vec<ItemParams> = (0..j).map(|i| ItemParams {
            a_p: (0..p).map(|d| if d == 0 { [0.9, 1.3, 0.7, 1.1][i % 4] }
                else if wording.contains(&i) { 0.9 } else { 0.0 }).collect(),
            a_s: (sm[i] >= 0).then_some([1.1, 0.8, 1.4, 1.0][i % 4]),
            d: vec![1.2, 0.0, -1.2],
        }).collect();
        let mut mask = vec![true; n * j];
        if case > 0 {
            mask[2 * j..3 * j].fill(false);
            if case == 1 {
                mask[j + 4..j + 8].fill(false);
                mask[1] = false;
                y[1] = usize::MAX; // masked sentinel는 category나 counts index가 되지 않는다.
                params[1].a_p[0] = -params[1].a_p[0];
            }
        }
        let observed = (case > 0).then_some(mask.as_slice());
        let v = validate_data(&y, observed, &pm, &sm, n, j, p, s, k, &cfg, false).unwrap();
        let (coords, log_w) = build_primary_grid(tz, wz, p, v.grid_size);
        let mut gh_digest = sha2::Sha256::new();
        for values in [&coords[..], &log_w[..], &log_ws[..]] {
            gh_digest.update((values.len() as u64).to_le_bytes());
            for value in values { gh_digest.update(value.to_le_bytes()); }
        }
        let gh_hash = format!("{:x}", gh_digest.finalize());
        let nodes: usize = v.item_block.iter().map(|b| v.grid_size * if b.is_some() { qs } else { 1 }).sum();
        let count_bytes = nodes.checked_mul(k).unwrap().checked_mul(8).unwrap();
        let count_headers = (nodes + j + 1).checked_mul(std::mem::size_of::<Vec<f64>>()).unwrap();
        let output = v.grid_size + s * v.grid_size * qs;
        let estimate = 6 * (count_bytes + count_headers) + 3 * count_bytes + 8 * output * n * 8 + 65536;
        assert!(estimate as u64 <= host_budget, "counts contract source estimate exceeds host budget");
        let input = json!({"case":case,"responses":y,"observed":observed,"primary_map":pm,
            "specific_map":sm,"n_persons":n,"n_items":j,"n_primary":p,"n_specific":s,"n_cat":k,
            "q_primary":qp,"q_specific":qs,"phi":"identity",
            "a_primary":params.iter().map(|x| x.a_p.as_slice()).collect::<Vec<_>>(),
            "a_specific":params.iter().map(|x| x.a_s).collect::<Vec<_>>(),
            "thresholds":params.iter().map(|x| x.d.as_slice()).collect::<Vec<_>>()});
        let input_hash = format!("{:x}", sha2::Sha256::digest(serde_json::to_vec(&input).unwrap()));
        let tables = item_logprob_tables(&v, &params, &coords, ts, v.grid_size).unwrap();
        let streamed = e_step_with_moments(&v, &y, observed, &params, &log_w, &log_ws,
            &coords, ts, v.grid_size, qs, None, true, None);
        let cached = e_step_with_moments(&v, &y, observed, &params, &log_w, &log_ws,
            &coords, ts, v.grid_size, qs, None, true, Some(&tables));
        assert_eq!(streamed.0.to_bits(), cached.0.to_bits());
        assert_eq!(streamed.1, cached.1);
        assert_eq!(streamed.2, cached.2);
        let mut stats = ReducedFitStatistics { counts: Vec::new(), cross: Vec::new() };
        let ll = e_step_gpu_person_moments(&v, &y, observed, &params, &log_w, &log_ws,
            &coords, ts, None, Some(&mut stats), true, gpu_budget).unwrap();
        assert_eq!(ll.to_bits(), cached.0.to_bits());
        assert_eq!(stats.counts.len(), j);
        assert_eq!(stats.cross.len(), p * p);
        let mut count_gap = 0.0_f64;
        let mut mass_residual = 0.0_f64;
        for i in 0..j {
            let li = v.grid_size * if v.item_block[i].is_some() { qs } else { 1 };
            assert_eq!(stats.counts[i].len(), li);
            for node in 0..li {
                assert_eq!(stats.counts[i][node].len(), k);
                for cat in 0..k {
                    let a = stats.counts[i][node][cat];
                    let b = cached.1[i][node][cat];
                    assert!(a.is_finite() && a >= 0.0);
                    assert_eq!(a.to_bits(), b.to_bits());
                    count_gap = count_gap.max((a - b).abs());
                }
            }
            for cat in 0..k {
                let actual: f64 = stats.counts[i].iter().map(|r| r[cat]).sum();
                let expected = (0..n).filter(|&pp| observed.is_none_or(|m| m[pp * j + i]) && y[pp * j + i] == cat).count();
                mass_residual = mass_residual.max((actual - expected as f64).abs());
            }
        }
        for (a, b) in stats.cross.iter().zip(&cached.2) { assert_eq!(a.to_bits(), b.to_bits()); }
        assert!(mass_residual.is_finite() && mass_residual <= 1e-3);
        let members = v.blocks.iter().map(Vec::len).sum::<usize>();
        let fixed = (8 + 2 * nodes * k + 2 * j + s + 1 + members.max(2) + 2 * v.grid_size + 2 * qs) as u64 * 4;
        let per_person = (j + 4 * v.grid_size + 4 * (s * v.grid_size * qs).max(1)) as u64 * 4;
        let short_budget = fixed + 3 * per_person;
        assert!(short_budget <= gpu_budget);
        let mut short = ReducedFitStatistics { counts: Vec::new(), cross: Vec::new() };
        let short_ll = e_step_gpu_person_moments(&v, &y, observed, &params, &log_w, &log_ws,
            &coords, ts, None, Some(&mut short), true, short_budget).unwrap();
        assert_eq!(short_ll.to_bits(), ll.to_bits());
        assert_eq!(short.counts, stats.counts);
        assert_eq!(short.cross, stats.cross);
        let original_params = params.clone();
        params[0].d[0] += 0.125; // M-step이 아니라 별도로 선언한 다음 고정 state다.
        let fresh_input = json!({"initial_input_sha256":input_hash,
            "thresholds":params.iter().map(|x| x.d.as_slice()).collect::<Vec<_>>()});
        let fresh_hash = format!("{:x}", sha2::Sha256::digest(serde_json::to_vec(&fresh_input).unwrap()));
        let fresh = e_step_with_moments(&v, &y, observed, &params, &log_w, &log_ws,
            &coords, ts, v.grid_size, qs, None, true, None);
        let mut next = ReducedFitStatistics { counts: Vec::new(), cross: Vec::new() };
        let next_ll = e_step_gpu_person_moments(&v, &y, observed, &params, &log_w, &log_ws,
            &coords, ts, None, Some(&mut next), true, gpu_budget).unwrap();
        assert_eq!(next_ll.to_bits(), fresh.0.to_bits());
        assert_eq!(next.counts, fresh.1);
        assert_eq!(next.cross, fresh.2);
        assert_ne!(next.counts, stats.counts);
        if item_update {
            fixed_item_return_contract(case, &v, &original_params, &coords, ts,
                &cached.1, &stats.counts, &input_hash);
        }
        println!("{}", json!({"event":"counts_case","input_sha256":input_hash,"input":input,
            "adapter_name":ctx.adapter_info.name,"adapter_backend":format!("{:?}",ctx.adapter_info.backend),
            "gpu_budget_bytes":gpu_budget,"short_batch_budget_bytes":short_budget,
            "host_budget_bytes":host_budget,"host_source_estimate_bytes":estimate,
            "host_budget_semantics":"source-shaped payload/header estimate; not RSS cap",
            "gh_coords_prior_sha256":gh_hash,"fresh_input":fresh_input,"fresh_input_sha256":fresh_hash,
            "counts_max_abs_delta":count_gap,"cross_bits_equal":true,"likelihood_bits_equal":true,
            "mass_max_abs_residual":mass_residual,"batch_contract":"3_then_1","fresh_params_checked":true,
            "fit_executed":false,"m_step_executed":item_update}));
        positive_cases += 1;
        if case == 0 {
            let mut bad_y = y.clone(); bad_y[0] = k;
            let mut bad_map = sm.clone(); bad_map[0] = s as i32;
            let short_mask = vec![true; n * j - 1];
            let rejects = [
                validate_data(&bad_y, None, &pm, &sm, n,j,p,s,k,&cfg,false).is_err(),
                validate_data(&y[..y.len()-1], None, &pm,&sm,n,j,p,s,k,&cfg,false).is_err(),
                validate_data(&y, Some(&short_mask), &pm,&sm,n,j,p,s,k,&cfg,false).is_err(),
                validate_data(&y, None, &pm[..pm.len()-1],&sm,n,j,p,s,k,&cfg,false).is_err(),
                validate_data(&y, None, &pm,&bad_map,n,j,p,s,k,&cfg,false).is_err(),
            ];
            assert!(rejects.iter().all(|&x| x));
            rejected_cases += rejects.len();
            let mut rejected = ReducedFitStatistics { counts: Vec::new(), cross: Vec::new() };
            assert!(e_step_gpu_person_moments(&v,&y,None,&params,&log_w,&log_ws,&coords,ts,
                None,Some(&mut rejected),true,fixed + per_person - 1).is_err());
            rejected_cases += 1;
            let loss_y = vec![0; n * j];
            let mut loss_params = params.clone();
            for par in &mut loss_params { par.d = vec![1e308, 0.0, -1e308]; }
            let cpu_loss = e_step_with_moments(&v,&loss_y,None,&loss_params,&log_w,&log_ws,
                &coords,ts,v.grid_size,qs,None,true,None).0;
            assert!(!cpu_loss.is_finite());
            let error = e_step_gpu_person_moments(&v,&loss_y,None,&loss_params,&log_w,&log_ws,
                &coords,ts,None,Some(&mut rejected),true,gpu_budget).unwrap_err();
            assert!(error.contains("nonfinite GPU-derived f64 likelihood"));
            rejected_cases += 1;
            println!("{}",json!({"event":"counts_rejections","validation_cases":rejects.len(),
                "validation_case_names":["observed_category","response_shape","mask_shape","primary_map_shape","specific_map_range"],
                "validation_rejected":rejects,"insufficient_budget_bytes":fixed + per_person - 1,
                "insufficient_budget_rejected":true,"finite_input_overflow_mass_loss_rejected":true,
                "mass_loss_thresholds":[1e308,0.0,-1e308],"mass_loss_responses_all_zero":true,
                "cpu_mass_loss_likelihood_finite":cpu_loss.is_finite(),"mass_loss_error":error,
                "fit_executed":false,"m_step_executed":false}));
        }
    }
    assert_eq!(positive_cases, 3);
    assert_eq!(rejected_cases, 7);
    println!("counts_contract_positive_cases={positive_cases}, rejected_cases={rejected_cases}, fit=false, m_step={item_update}");
}

/// 기본 모드는 고정 문항은행의 같은 GH 노드·가중치에서 1인의 적률을 대조한다.
/// 명시적 counts 모드는 작은 4인 고정 state의 통계만 검사한다.
/// 별도 itemupdate 모드는 그 고정 counts의 대표 item 반환을 대조한다.
/// host 예산은 소스 유래 보수 추정 정책이며 실제 RSS 강제 상한이 아니다.
/// 부모 작업에서 확인한 Cai (2010, pp. 608–609, Appendices A/B)의 사후
/// 1·2차 적률이 근거다. 자료 생성·모수 갱신·수렴·회복도 검증은 하지 않는다.
/// 참고문헌: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. *Psychometrika, 75*(4), 581–612.
/// https://doi.org/10.1007/s11336-010-9178-0.
#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
#[ignore = "requires hardware GPU; fixed-bank E-step kernel profile only"]
fn reference_gpu_fixed_bank_kernel_profile() {
    use crate::two_tier_grm::{
        build_primary_grid, e_step_gpu_person_moments, e_step_with_moments, gh_rule,
        item_logprob_tables, validate_data, ItemParams, LatentPosteriorMoments,
    };
    use serde_json::json;
    use sha2::Digest;
    use std::time::Instant;

    let required_usize = |name: &str| -> usize {
        std::env::var(name)
            .unwrap_or_else(|_| panic!("required environment variable {name} is missing"))
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("environment variable {name} must be an unsigned integer: {e}"))
    };
    let required_u64 = |name: &str| -> u64 {
        std::env::var(name)
            .unwrap_or_else(|_| panic!("required environment variable {name} is missing"))
            .parse::<u64>()
            .unwrap_or_else(|e| panic!("environment variable {name} must be an unsigned integer: {e}"))
    };
    let q_primary = required_usize("G1_KERNEL_Q_PRIMARY");
    let q_specific = required_usize("G1_KERNEL_Q_SPECIFIC");
    let gpu_budget = required_u64("G1_KERNEL_GPU_BUDGET_BYTES");
    let host_budget = required_u64("G1_KERNEL_HOST_BUDGET_BYTES");
    assert!(q_primary > 0 && q_specific > 0, "노드 수는 양수여야 합니다");
    assert!(gpu_budget > 0 && host_budget > 0, "메모리 예산은 양수여야 합니다");
    let execution_head = std::env::var("G1_EXECUTION_HEAD").unwrap_or_else(|_| "unknown".into());
    println!("{}", json!({
        "event": "source", "execution_head": execution_head,
        "numerical_source": "crates/mlsirm-core/src/two_tier_grm.rs",
        "numerical_source_sha256": format!("{:x}", sha2::Sha256::digest(include_bytes!("../../crates/mlsirm-core/src/two_tier_grm.rs"))),
        "kernel_source": "crates/mlsirm-core/src/gpu_bifactor.rs",
        "kernel_source_sha256": format!("{:x}", sha2::Sha256::digest(include_bytes!("../../crates/mlsirm-core/src/gpu_bifactor.rs"))),
        "test_source": "tests/unit/two_tier_grm_tests.rs",
        "test_source_sha256": format!("{:x}", sha2::Sha256::digest(include_bytes!("two_tier_grm_tests.rs"))),
    }));

    let counts_contract = match std::env::var("G1_KERNEL_COUNTS_CONTRACT") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        _ => panic!("G1_KERNEL_COUNTS_CONTRACT must be absent, 0 or 1"),
    };
    let itemupdate_contract = match std::env::var("G1_KERNEL_ITEMUPDATE_CONTRACT") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        _ => panic!("G1_KERNEL_ITEMUPDATE_CONTRACT must be absent, 0 or 1"),
    };
    let candidate_contract = match std::env::var("G1_KERNEL_CANDIDATE_CONTRACT") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        _ => panic!("G1_KERNEL_CANDIDATE_CONTRACT must be absent, 0 or 1"),
    };
    assert!(usize::from(counts_contract)+usize::from(itemupdate_contract)+usize::from(candidate_contract)<=1,
        "select only one fixed-state contract mode");
    if candidate_contract {
        fixed_item_candidate_contract();
        return;
    }
    if counts_contract || itemupdate_contract {
        reference_gpu_fixed_bank_counts_contract(q_primary, q_specific, gpu_budget, host_budget, itemupdate_contract);
        return;
    }

    // Fixed known parameters mirror the continuous P=2, S=4 fixture bank;
    // this is a pointwise E-step profile, not a fit or recovery acceptance.
    // Cai (2010), pp. 608–609, Appendices A/B: the posterior moments are
    // conditional on a fixed item bank and supplied quadrature nodes/weights.
    let n_items = 16usize;
    let n_primary = 2usize;
    let n_specific = 4usize;
    let n_cat = 4usize;
    const RESPONSES: [usize; 16] = [0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2];
    const WORDING: [usize; 7] = [4, 5, 6, 9, 12, 14, 15];
    const PRIMARY_LOADINGS: [f64; 4] = [0.9, 1.3, 0.7, 1.1];
    const SPECIFIC_LOADINGS: [f64; 4] = [1.1, 0.8, 1.4, 1.0];
    const THRESHOLDS: [f64; 3] = [1.2, 0.0, -1.2];
    let y = RESPONSES.to_vec();
    let primary_map: Vec<bool> = (0..n_items)
        .flat_map(|i| [true, WORDING.contains(&i)])
        .collect();
    let specific_map: Vec<i32> = (0..n_items).map(|i| (i / 4) as i32).collect();
    let params: Vec<ItemParams> = (0..n_items)
        .map(|i| ItemParams {
            a_p: vec![PRIMARY_LOADINGS[i % 4], if WORDING.contains(&i) { 0.9 } else { 0.0 }],
            a_s: Some(SPECIFIC_LOADINGS[i % 4]),
            d: THRESHOLDS.to_vec(),
        })
        .collect();
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        q_primary,
        q_specific,
        ..valid_config()
    };
    let validated = validate_data(
        &y, None, &primary_map, &specific_map, 1, n_items, n_primary, n_specific,
        n_cat, &cfg, false,
    ).expect("fixed P=2/S=4 reference bank must validate");
    let grid = q_primary.checked_mul(q_primary).expect("primary grid size overflows usize");
    let table_elements = validated.item_block.iter().try_fold(0usize, |sum, block| {
        grid.checked_mul(if block.is_some() { q_specific } else { 1 })
            .and_then(|len| len.checked_mul(n_cat))
            .and_then(|len| sum.checked_add(len))
    }).expect("fixed item table dimensions overflow usize");
    let joint_elements = n_specific.checked_mul(grid)
        .and_then(|len| len.checked_mul(q_specific))
        .expect("fixed raw joint posterior dimensions overflow usize");
    // Upper-bound expected-count payload even though this profile never collects it.
    let count_elements = table_elements;
    let checked_bytes = |elements: usize, width: usize| -> u64 {
        u64::try_from(elements.checked_mul(width).expect("host estimate overflows usize"))
            .expect("host estimate overflows u64")
    };
    let add_bytes = |a: u64, b: u64| a.checked_add(b).expect("host estimate overflows u64");
    let table_headers = checked_bytes(2 * (n_items + 1), std::mem::size_of::<Vec<f64>>());
    let output_elements = grid.checked_add(joint_elements).expect("output estimate overflows usize");
    let count_nodes = validated.item_block.iter().try_fold(0usize, |sum, block| {
        grid.checked_mul(if block.is_some() { q_specific } else { 1 })
            .and_then(|len| sum.checked_add(len))
    }).expect("conservative count header estimate overflows usize");
    let count_header_count = n_items.checked_add(count_nodes).and_then(|n| n.checked_add(1))
        .expect("conservative count header estimate overflows usize");
    let count_headers = checked_bytes(count_header_count, std::mem::size_of::<Vec<f64>>());
    let latent_count = n_primary + n_specific;
    let host_estimate = [
        checked_bytes(table_elements, std::mem::size_of::<f64>()), // retained CPU cached reference GRM table
        checked_bytes(table_elements, std::mem::size_of::<f64>()), // GPU E-step host GRM table
        checked_bytes(table_elements, std::mem::size_of::<[u32; 2]>()), // conservative reference-table upload staging copy
        checked_bytes(n_items, std::mem::size_of::<usize>()), // fixed response bank
        checked_bytes(n_items, std::mem::size_of::<i32>()), // GPU response staging
        checked_bytes(output_elements, std::mem::size_of::<f64>()), // decoded raw GPU products/readback
        checked_bytes(output_elements.checked_mul(2).expect("readback staging estimate overflows usize"), std::mem::size_of::<f32>()), // two f32 words per f64 readback value
        checked_bytes(output_elements, std::mem::size_of::<f64>()), // posterior arrays
        checked_bytes(6 * latent_count, std::mem::size_of::<f64>()), // CPU/GPU moments and both marginal SD arrays
        checked_bytes(grid.checked_mul(n_primary + 2)
            .and_then(|len| len.checked_add(q_primary))
            .and_then(|len| q_specific.checked_mul(n_specific + 1).and_then(|n| len.checked_add(n)))
            .expect("coordinate/weight estimate overflows usize"), std::mem::size_of::<f64>()), // shared coords/log weights, GPU group-node copies, build scratch
        checked_bytes(grid.checked_add(q_specific).expect("prior word estimate overflows usize"), std::mem::size_of::<[u32; 2]>()), // GPU state prior-word copies
        checked_bytes(n_specific.checked_mul(grid).and_then(|len| len.checked_add(grid))
            .expect("GPU normalization scratch estimate overflows usize"), std::mem::size_of::<f64>()),
        checked_bytes(n_specific.checked_mul(grid)
            .and_then(|len| grid.checked_mul(3).and_then(|scratch| len.checked_add(scratch)))
            .and_then(|len| len.checked_add(q_specific))
            .and_then(|len| n_specific.checked_mul(q_specific).and_then(|scratch| len.checked_add(scratch)))
            .and_then(|len| n_primary.checked_mul(n_primary).and_then(|scratch| len.checked_add(scratch)))
            .expect("source-shaped E-step scratch estimate overflows usize"), std::mem::size_of::<f64>()),
        checked_bytes(count_elements, std::mem::size_of::<f64>()), // conservative counts, not collected
        table_headers,
        count_headers,
        checked_bytes(40, std::mem::size_of::<Vec<f64>>()), // scratch, group nodes, raw/readback/posterior/moment and state Vec headers
        checked_bytes(3, std::mem::size_of::<Vec<Vec<f64>>>()),
        checked_bytes(8 + n_items * 4 + n_specific + 1, std::mem::size_of::<u32>()), // state dims/offsets/block IDs/members, including member capacity
        checked_bytes(n_items, std::mem::size_of::<ItemParams>()),
        checked_bytes(n_items * (n_primary + n_cat - 1), std::mem::size_of::<f64>()), // fixed bank slopes/thresholds
        checked_bytes(n_items * n_primary, std::mem::size_of::<bool>()),
        checked_bytes(n_items, std::mem::size_of::<i32>()), // fixed specific map
        checked_bytes(n_items * (n_primary + n_specific + n_cat), std::mem::size_of::<usize>()), // conservative validated map/category payload and capacities
        checked_bytes(n_items, std::mem::size_of::<Option<usize>>()),
        checked_bytes(2 * n_items + n_specific, std::mem::size_of::<Vec<usize>>()), // validated map headers
        checked_bytes(1, std::mem::size_of_val(&validated)),
        checked_bytes(n_items * (n_primary + n_cat + 6) + 32, std::mem::size_of::<serde_json::Value>()), // input JSON tree envelope, emitted and dropped before E-steps
        checked_bytes(n_items * (n_primary + n_cat + 6) + 32, 4 * 64), // fixed-bank JSON text/key envelope and serialization capacity; <=64 bytes per scalar/key

    ].into_iter().fold(0u64, add_bytes);
    assert!(host_estimate <= host_budget,
        "declared fixed-node host estimate {host_estimate} bytes exceeds G1_KERNEL_HOST_BUDGET_BYTES={host_budget}; refusing to reduce nodes");

    let context = crate::gpu::GpuContext::get().expect("hardware GPU context must initialize");
    assert!(matches!(context.adapter_info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu),
        "reference GPU kernel profile requires a hardware adapter, got {:?}", context.adapter_info.device_type);
    let limits = context.device.limits();
    let block_members = validated.blocks.iter().map(Vec::len).sum::<usize>();
    let fixed_elements = [
        8usize,
        table_elements.checked_mul(2).expect("GPU policy table estimate overflows usize"),
        n_items,
        n_items,
        n_specific.checked_add(1).expect("GPU policy dimensions overflow usize"),
        block_members.max(2),
        grid.checked_mul(2).expect("GPU policy grid estimate overflows usize"),
        q_specific.checked_mul(2).expect("GPU policy specific estimate overflows usize"),
    ].into_iter().try_fold(0usize, |sum, len| sum.checked_add(len))
        .expect("GPU policy fixed estimate overflows usize");
    let policy_joint = n_specific.checked_mul(grid).and_then(|len| len.checked_mul(q_specific))
        .expect("GPU policy joint estimate overflows usize");
    let person_elements = [n_items, grid.checked_mul(4).expect("GPU policy primary estimate overflows usize"),
        policy_joint.max(1).checked_mul(4).expect("GPU policy joint estimate overflows usize")]
        .into_iter().try_fold(0usize, |sum, len| sum.checked_add(len))
        .expect("GPU policy person estimate overflows usize");
    let gpu_policy_bytes = checked_bytes(
        fixed_elements.checked_add(person_elements).expect("GPU policy budget estimate overflows usize"),
        std::mem::size_of::<u32>(),
    );
    assert!(gpu_budget >= gpu_policy_bytes,
        "G1_KERNEL_GPU_BUDGET_BYTES={gpu_budget} is below the existing reference_precision one-person source policy requirement {gpu_policy_bytes}; refusing to reduce declared nodes");
    let binding_bytes_per_person = policy_joint.max(n_items).max(grid)
        .checked_mul(std::mem::size_of::<f64>()).expect("GPU binding estimate overflows usize") as u64;
    let binding_limit = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    assert!(binding_bytes_per_person as u64 <= binding_limit,
        "one-person binding estimate {binding_bytes_per_person} exceeds current adapter limit {binding_limit}");
    println!("{{\"event\":\"resource\",\"adapter_name\":{},\"adapter_backend\":{},\"adapter_type\":{},\"max_storage_buffer_binding_bytes\":{},\"max_buffer_bytes\":{},\"max_storage_buffers_per_stage\":{},\"max_workgroups_per_dimension\":{},\"host_estimate_bytes\":{},\"host_budget_bytes\":{},\"host_budget_semantics\":\"source-derived conservative allocation estimate; not RSS cap\",\"gpu_policy_minimum_bytes\":{},\"gpu_budget_bytes\":{},\"gpu_budget_semantics\":\"source policy; adapter limits are per-buffer constraints, not available physical VRAM\",\"execution_head\":{}}}",
        serde_json::to_string(&context.adapter_info.name).unwrap(),
        serde_json::to_string(&format!("{:?}", context.adapter_info.backend)).unwrap(),
        serde_json::to_string(&format!("{:?}", context.adapter_info.device_type)).unwrap(), limits.max_storage_buffer_binding_size,
        limits.max_buffer_size, limits.max_storage_buffers_per_shader_stage,
        limits.max_compute_workgroups_per_dimension, host_estimate, host_budget, gpu_policy_bytes, gpu_budget,
        serde_json::to_string(&execution_head).unwrap());

    let (primary_nodes, primary_weights) = gh_rule(q_primary).expect("primary quadrature nodes must be available");
    let (specific_nodes, specific_weights) = gh_rule(q_specific).expect("specific quadrature nodes must be available");
    let (coords, log_w) = build_primary_grid(primary_nodes, primary_weights, n_primary, grid);
    let log_ws: Vec<f64> = specific_weights.iter().map(|w| w.ln()).collect();
    let input = json!({
        "responses": y, "primary_map": primary_map, "specific_map": specific_map,
        "primary_loadings": params.iter().map(|p| &p.a_p).collect::<Vec<_>>(),
        "specific_loadings": params.iter().map(|p| p.a_s).collect::<Vec<_>>(),
        "thresholds": params.iter().map(|p| &p.d).collect::<Vec<_>>(),
        "n_persons": 1, "n_items": n_items, "n_primary": n_primary,
        "n_specific": n_specific, "n_cat": n_cat, "q_primary": q_primary,
        "q_specific": q_specific, "phi": "identity"
    });
    let input_bytes = serde_json::to_vec(&input).unwrap();
    println!("{{\"event\":\"input\",\"input_sha256\":\"{:x}\",\"seed\":null,\"input_json\":{}}}",
        sha2::Sha256::digest(&input_bytes), String::from_utf8(input_bytes).unwrap());
    drop(input);

    REFERENCE_GPU_TIMINGS.with(|timings| *timings.borrow_mut() = Some(Vec::new()));
    let cpu_tables_started = Instant::now();
    let tables = item_logprob_tables(&validated, &params, &coords, specific_nodes, grid).unwrap();
    let cpu_table_seconds = cpu_tables_started.elapsed().as_secs_f64();
    let mut cpu_moments = LatentPosteriorMoments {
        mean: vec![0.0; latent_count], second: vec![0.0; latent_count],
    };
    let cpu_started = Instant::now();
    let (cpu_ll, _, _) = e_step_with_moments(
        &validated, &y, None, &params, &log_w, &log_ws, &coords, specific_nodes,
        grid, q_specific, Some(&mut cpu_moments), false, Some(&tables),
    );
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();
    let mut gpu_moments = LatentPosteriorMoments {
        mean: vec![0.0; latent_count], second: vec![0.0; latent_count],
    };
    let gpu_started = Instant::now();
    let gpu_ll = e_step_gpu_person_moments(
        &validated, &y, None, &params, &log_w, &log_ws, &coords, specific_nodes,
        Some(&mut gpu_moments), None, true, gpu_budget,
    ).expect("the declared-node reference GPU E-step must succeed without CPU fallback");
    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
    let timings = REFERENCE_GPU_TIMINGS.with(|timings| timings.borrow_mut().take().unwrap_or_default());
    let required_timing_keys = [
        "table_generation_seconds", "gpu_state_preparation_seconds",
        "gpu_log_products_and_readback_seconds", "rust_lse_exp_normalization_seconds",
        "posterior_moment_contraction_seconds", "f64_likelihood_certification_seconds",
        "prepare_validation_metadata_seconds", "prepare_error_scope_setup_seconds",
        "prepare_uniform_buffer_seconds", "prepare_table_map_copy_seconds",
        "prepare_input_output_buffers_seconds", "prepare_layout_bindgroup_seconds",
        "prepare_shader_module_seconds", "prepare_compute_pipeline_seconds",
        "prepare_error_receipt_seconds", "sweep_queue_write_encode_seconds",
        "sweep_readback_buffer_preparation_seconds", "sweep_submit_map_decode_seconds",
        "sweep_error_receipt_seconds",
    ];
    let validate_timing_receipt = |timings: &[(&str, f64)]| {
        if timings.len() != required_timing_keys.len() {
            return Err("필수 단계 시간 기록이 누락됐습니다");
        }
        for key in required_timing_keys {
            if timings.iter().filter(|(name, _)| *name == key).count() != 1 {
                return Err("각 시간 키를 정확히 한 번 기록해야 합니다");
            }
        }
        if timings.iter().any(|(_, seconds)| !seconds.is_finite() || *seconds < 0.0) {
            return Err("단계 시간은 유한하고 비음수여야 합니다");
        }
        Ok(())
    };
    validate_timing_receipt(&timings).expect("완전하고 유효한 단계 시간 receipt가 필요합니다");
    let valid: Vec<_> = required_timing_keys.iter().map(|key| (*key, 0.0)).collect();
    assert!(validate_timing_receipt(&valid).is_ok());
    let mut missing = valid.clone();
    missing.pop();
    assert!(validate_timing_receipt(&missing).is_err());
    let mut duplicated = valid.clone();
    duplicated[1].0 = duplicated[0].0;
    assert!(validate_timing_receipt(&duplicated).is_err());
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        let mut invalid_values = valid.clone();
        invalid_values[0].1 = invalid;
        assert!(validate_timing_receipt(&invalid_values).is_err());
    }
    println!("timing_receipt_negative_cases=5: 누락·중복·NaN·무한대·음수 거부 확인");
    for seconds in [cpu_seconds, gpu_seconds, cpu_table_seconds] {
        assert!(seconds.is_finite() && seconds >= 0.0, "유효하지 않은 전체 시간입니다");
    }
    for moments in [&cpu_moments, &gpu_moments] {
        assert_eq!(moments.mean.len(), latent_count);
        assert_eq!(moments.second.len(), latent_count);
    }
    assert!(cpu_ll.is_finite() && gpu_ll.is_finite(), "CPU/GPU likelihoods must be finite");
    let max_delta = |left: &[f64], right: &[f64]| {
        assert_eq!(left.len(), right.len(), "CPU/GPU posterior output shapes must agree");
        assert!(left.iter().chain(right).all(|v| v.is_finite()), "CPU/GPU posterior outputs must be finite");
        left.iter().zip(right).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    };
    let ll_delta = (cpu_ll - gpu_ll).abs();
    let mean_delta = max_delta(&cpu_moments.mean, &gpu_moments.mean);
    let second_delta = max_delta(&cpu_moments.second, &gpu_moments.second);
    let sd = |moments: &LatentPosteriorMoments| moments.mean.iter().zip(&moments.second)
        .map(|(mean, second)| {
            let variance = second - mean * mean;
            assert!(variance.is_finite() && variance >= -1e-12 * second.abs().max(1.0),
                "posterior marginal variance must be finite and nonnegative apart from roundoff");
            variance.max(0.0).sqrt()
        }).collect::<Vec<_>>();
    let sd_delta = max_delta(&sd(&cpu_moments), &sd(&gpu_moments));
    println!("{{\"event\":\"timing\",\"whole_cpu_estep_seconds\":{cpu_seconds},\"whole_gpu_estep_seconds\":{gpu_seconds},\"cpu_table_generation_seconds\":{cpu_table_seconds},\"whole_cpu_including_table_seconds\":{},\"stages\":{{{}}},\"note\":\"host clock with clock/collector overhead; child stages subdivide parent intervals and must not be double-counted; driver lazy work may occur during submit; no GPU timestamp or isolated PCIe timing claim\"}}",
        cpu_seconds + cpu_table_seconds,
        timings.iter().map(|(name, seconds)| format!("{}:{}", serde_json::to_string(name).unwrap(), seconds))
            .collect::<Vec<_>>().join(","));
    println!("{{\"event\":\"comparison\",\"loglik_abs_delta\":{ll_delta},\"mean_max_abs_delta\":{mean_delta},\"second_max_abs_delta\":{second_delta},\"marginal_sd_max_abs_delta\":{sd_delta},\"tolerance\":0.001,\"collect_counts\":false,\"parameter_updates\":false}} ");
    assert!(ll_delta < 1e-3, "CPU/GPU fixed-bank likelihood gap {ll_delta} exceeds unchanged 1e-3 regression bound");
    assert!(mean_delta < 1e-3, "CPU/GPU posterior mean gap {mean_delta} exceeds unchanged 1e-3 regression bound");
    assert!(second_delta < 1e-3, "CPU/GPU posterior second-moment gap {second_delta} exceeds unchanged 1e-3 regression bound");
    assert!(sd_delta < 1e-3, "CPU/GPU posterior marginal SD gap {sd_delta} exceeds unchanged 1e-3 regression bound");
}

#[test]
fn reduced_marginal_loglik_matches_brute_force_product_grid() {
    let (a_p, a_s, thresholds, phi) = tiny_params();
    let (y, n_persons) = tiny_data();
    let reduced = two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &phi,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect("reduced loglik on valid tiny input must succeed");
    let brute = two_tier_grm_marginal_loglik_brute(
        &a_p,
        &a_s,
        &thresholds,
        &phi,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect("brute-force loglik on valid tiny input must succeed");
    assert!(
        reduced.is_finite() && brute.is_finite(),
        "both logliks must be finite: reduced={reduced}, brute={brute}"
    );
    let gap = (reduced - brute).abs();
    assert!(
        gap <= 1e-10,
        "two-tier reduction only reorders the finite marginal sum, so the gap \
         must be floating-point noise (<= 1e-10); got gap={gap:.3e} \
         (reduced={reduced:.10}, brute={brute:.10})"
    );
}

#[test]
fn single_primary_oracle_matches_stage1_bifactor_oracle() {
    use crate::bifactor_grm::bifactor_grm_marginal_loglik;
    // P = 1 with every item on the single primary IS the stage-1 bifactor
    // problem (4 items, 2 specifics, 3 categories; the stage-1 unit layout).
    let a_general = vec![1.2, -0.9, 1.0, 0.7];
    let a_specific = vec![0.8, 1.1, 0.9, 0.6];
    let thresholds = vec![0.9, -0.7, 0.5, -1.0, 1.1, -0.4, 0.2, -1.3];
    let specific_map = [0, 0, 1, 1];
    let primary_map = [true; 4];
    let y = vec![
        0, 1, 2, 0, //
        1, 2, 0, 1, //
        2, 0, 1, 2, //
        0, 2, 1, 0, //
        1, 1, 1, 1, //
        2, 2, 2, 2, //
        0, 0, 0, 0, //
        2, 1, 0, 2, //
        1, 0, 2, 1, //
        0, 2, 0, 1, //
        1, 2, 1, 0, //
        2, 0, 2, 1,
    ];
    let n_persons = 12usize;
    let phi = vec![1.0];
    // Row-major 4x1 primary slopes equal the general slopes.
    let two_tier = two_tier_grm_marginal_loglik(
        &a_general,
        &a_specific,
        &thresholds,
        &phi,
        &y,
        None,
        &primary_map,
        &specific_map,
        n_persons,
        4,
        1,
        2,
        3,
        7,
        7,
    )
    .expect("two-tier P=1 oracle must succeed");
    let stage1 = bifactor_grm_marginal_loglik(
        &a_general,
        &a_specific,
        &thresholds,
        &y,
        None,
        &specific_map,
        n_persons,
        4,
        2,
        3,
        7,
        7,
    )
    .expect("stage-1 oracle must succeed");
    let gap = (two_tier - stage1).abs();
    assert!(
        gap <= 1e-10,
        "P=1 two-tier IS the bifactor model, so both oracles evaluate the same \
         finite sum; got gap={gap:.3e} (two-tier={two_tier:.10}, stage-1={stage1:.10})"
    );
}

// ---------------------------------------------------------------------------
// Argument validation and failure reporting.
// ---------------------------------------------------------------------------

fn valid_config() -> TwoTierGrmConfig {
    TwoTierGrmConfig {
        estimate_primary_correlation: true,
        q_primary: 7,
        q_specific: 7,
        max_iter: 5,
        tol: 1e-4,
        n_starts: 1,
        seed: 42,
        newton_iter: 3,
        ridge: 1e-8,
    }
}

#[test]
fn reference_fit_gpu_is_explicit_and_never_substituted() {
    let (y, n) = tiny_data();
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        ..valid_config()
    };
    let correlated = fit_two_tier_grm_with_device(
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
        Device::Gpu,
        Some(1 << 28),
    );
    assert!(correlated
        .unwrap_err()
        .contains("identity primary correlation"));
    let fit = |device, budget| {
        fit_two_tier_grm_with_device(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
            device,
            budget,
        )
    };
    assert!(fit(Device::Auto, None).unwrap_err().contains("cpu or gpu"));
    assert!(fit(Device::Gpu, None)
        .unwrap_err()
        .contains("gpu_memory_budget_bytes"));
    assert!(fit(Device::Gpu, Some(0))
        .unwrap_err()
        .contains("gpu_memory_budget_bytes"));
    // A one-byte budget cannot dispatch; no CPU fit may be returned instead.
    assert!(fit(Device::Gpu, Some(1)).is_err());
    let cpu = fit(Device::Cpu, None).unwrap();
    assert_eq!(cpu.backend, "cpu");
    assert!(cpu.gpu_adapter_name.is_none());
    assert!(cpu.gpu_adapter_backend.is_none());
}

#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
#[ignore = "requires hardware GPU; CPU/GPU reference-fit parity"]
fn reference_fit_actual_gpu_matches_cpu() {
    if std::env::var("G1_GPU_RESOURCE_PROBE").as_deref()==Ok("1") {
        let context=crate::gpu::GpuContext::get().expect("hardware GPU context must initialize");
        assert!(matches!(context.adapter_info.device_type,wgpu::DeviceType::DiscreteGpu
            |wgpu::DeviceType::IntegratedGpu|wgpu::DeviceType::VirtualGpu));
        let limits=context.device.limits();
        println!("resource_probe_only=true, adapter_name={}, adapter_backend={:?}, adapter_type={:?}, max_storage_binding_bytes={}, max_buffer_bytes={}, max_storage_buffers_per_stage={}, max_workgroups_per_dimension={}",
            context.adapter_info.name,context.adapter_info.backend,context.adapter_info.device_type,
            limits.max_storage_buffer_binding_size,limits.max_buffer_size,
            limits.max_storage_buffers_per_shader_stage,limits.max_compute_workgroups_per_dimension);
        println!("resource_probe_only: no model fit, no quadrature reduction, no numerical acceptance");
        return;
    }
    // GPU의 integer binary64 덧셈을 실제 CPU f64 bit와 먼저 대조한다.
    use crate::gpu_bifactor::{e_step_reduced_gpu_log_products, ReducedEstepInputs};
    let mut pairs = vec![
        (0.0, -0.0),
        (-0.0, -0.0),
        (-1e-320, -1e-308),
        (-1.0, -f64::EPSILON / 2.0),
        (-1.0, -f64::EPSILON * 1.5),
        (-1e200, -1e-200),
        (-f64::from_bits(1), -f64::from_bits(1)),
        (-f64::from_bits(0x000fffffffffffff), -f64::from_bits(1)),
        (-f64::MIN_POSITIVE, -f64::from_bits(1)),
        (-f64::MAX / 2.0, -f64::MAX / 2.0),
    ];
    pairs.extend([
        (f64::NEG_INFINITY, -1.0),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        (-0.0, f64::NEG_INFINITY),
    ]);
    let mut state = 20260930u64;
    for _ in 0..1024 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = f64::from_bits((state & 0x7fdfffffffffffff) | 0x8000000000000000);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = f64::from_bits((state & 0x7fdfffffffffffff) | 0x8000000000000000);
        pairs.push((a, b));
    }
    let tables = vec![vec![
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    ]];
    let prior = vec![0.0; pairs.len()];
    let gnodes = vec![prior.clone()];
    let snodes = vec![Vec::new()];
    let probe = ReducedEstepInputs {
        y: &[0, 0],
        observed: None,
        group_id: None,
        n_persons: 1,
        n_items: 2,
        n_specific: 0,
        n_cat: 1,
        qg: pairs.len(),
        qs: 1,
        n_groups: 1,
        tables_groups: &tables,
        item_block: &[None, None],
        blocks: &[],
        tg_groups: &gnodes,
        ts_groups: &snodes,
        log_wg: &prior,
        log_ws: &[0.0],
    };
    let (actual, block) =
        e_step_reduced_gpu_log_products(&probe).expect("hardware word addition must run");
    assert!(block.is_empty());
    for ((a, b), actual) in pairs.iter().zip(actual) {
        assert_eq!(
            (0.0 + a + b).to_bits(),
            actual.to_bits(),
            "GPU addition differs: {a}+{b}"
        );
    }
    println!("actual_gpu_binary64_addition_cases={} passed", pairs.len());
    // 한 번 준비한 table/pipeline을 full·short·missing batch 사이에 재사용한다.
    use crate::gpu_bifactor::GpuLogProductState;
    let batch_probe=ReducedEstepInputs {y:&[0,0,0,0,0,0],n_persons:3,..probe};
    let prepared=GpuLogProductState::new(&batch_probe).unwrap();
    let (full,_)=prepared.sweep(&[0,0,0,0,0,0],None,3).unwrap();
    assert_eq!(full.len(),3*pairs.len());
    let (short,_)=prepared.sweep(&[0,0],None,1).unwrap();
    assert_eq!(short.len(),pairs.len());
    assert_eq!(short,full[..pairs.len()]);
    let (missing,_)=prepared.sweep(&[0,0],Some(&[false,false]),1).unwrap();
    assert!(missing.iter().all(|v|*v==0.0));
    assert!(prepared.sweep(&[0,0],Some(&[false]),1).is_none());
    assert!(prepared.sweep(&[0,0,0,0,0,0,0,0],None,4).is_none());
    let mut changed=tables.clone();changed[0][0][0]=-3.0;
    let changed_probe=ReducedEstepInputs {tables_groups:&changed,..probe};
    let next_iteration=GpuLogProductState::new(&changed_probe).unwrap();
    assert_eq!(next_iteration.sweep(&[0,0],None,1).unwrap().0[0],-3.0);
    assert_eq!(prepared.sweep(&[0,0],None,1).unwrap().0[0],0.0);
    println!("actual_GPU_fixed_state_reuse_full_short_missing_invalid_new_iteration_cases=7 passed");
    for invalid in [f64::NAN, f64::INFINITY, 1e-300] {
        let mut rejected = tables.clone();
        rejected[0][0][0] = invalid;
        let input = ReducedEstepInputs {
            tables_groups: &rejected,
            ..probe
        };
        assert!(
            e_step_reduced_gpu_log_products(&input).is_none(),
            "NaN/positive-infinity/positive log input must reject"
        );
    }
    let mut overflow = tables.clone();
    overflow[0][0][0] = -f64::MAX;
    overflow[0][1][0] = -f64::MAX;
    let input = ReducedEstepInputs {
        tables_groups: &overflow,
        ..probe
    };
    let (overflow_result, _) = e_step_reduced_gpu_log_products(&input).unwrap();
    assert_eq!(overflow_result[0], f64::NEG_INFINITY);
    println!("actual_gpu_binary64_malformed_input_rejection_cases=3 passed; finite_overflow_zero_mass_node=1 preserved");
    // Cai (2010), pp.608-609 Appendices A/B; mixed-precision regression
    // bounds are implementation choices, not scientific acceptance thresholds.
    let (y, n) = tiny_data();
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        max_iter: 500,
        tol: 1e-4,
        ..valid_config()
    };
    let started = std::time::Instant::now();
    let cpu = fit_two_tier_grm(
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .unwrap();
    let cpu_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let gpu = fit_two_tier_grm_with_device(
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
        Device::Gpu,
        Some(1 << 28),
    )
    .unwrap();
    let gpu_seconds = started.elapsed().as_secs_f64();
    assert!(cpu.converged && gpu.converged);
    assert_eq!(gpu.backend, "gpu");
    assert!(gpu.gpu_adapter_name.as_ref().is_some_and(|s| !s.is_empty()));
    assert!(gpu
        .gpu_adapter_backend
        .as_ref()
        .is_some_and(|s| !s.is_empty()));
    assert_eq!(cpu.category_counts, gpu.category_counts);
    assert_eq!(cpu.n_parameters, gpu.n_parameters);
    for (name, left, right) in [
        ("a_primary", &cpu.a_primary, &gpu.a_primary),
        ("a_specific", &cpu.a_specific, &gpu.a_specific),
        ("threshold", &cpu.threshold, &gpu.threshold),
        ("phi", &cpu.phi, &gpu.phi),
        ("theta_p_eap", &cpu.theta_p_eap, &gpu.theta_p_eap),
        ("theta_p_sd", &cpu.theta_p_sd, &gpu.theta_p_sd),
    ] {
        let gap = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("{name}_max_delta={gap}");
        assert!(gap < 1e-3, "{name} gap: {gap}");
    }
    let gap = gpu.loglik_trace.last().unwrap() - cpu.loglik_trace.last().unwrap();
    assert!(gap.abs() < 1e-3, "likelihood/AIC/BIC gap: {gap}");
    use sha2::Digest;
    let input = serde_json::to_vec(&(
        &y,
        TINY_PRIMARY_MAP,
        TINY_SPECIFIC_MAP,
        n,
        (
            cfg.q_primary,
            cfg.q_specific,
            cfg.max_iter,
            cfg.tol,
            cfg.n_starts,
            cfg.seed,
            cfg.newton_iter,
            cfg.ridge,
            cfg.estimate_primary_correlation,
        ),
    ))
    .unwrap();
    println!("input_sha256={:x}", sha2::Sha256::digest(&input));
    println!("primary_correlation=identity, adapter={:?}, adapter_backend={:?}, cpu_iterations={}, gpu_iterations={}, cpu_seconds={cpu_seconds}, gpu_seconds={gpu_seconds}, loglik_delta={gap}, AIC_delta={}, BIC_delta={}",
            gpu.gpu_adapter_name, gpu.gpu_adapter_backend, cpu.n_iter, gpu.n_iter, -2.0*gap, -2.0*gap);

    // 합법적인 finite GRM의 0 범주확률/-inf 노드를 삭제하거나 floor로 대체하지 않는다.
    use crate::two_tier_grm::{validate_data, ItemParams, LatentPosteriorMoments};
    assert_eq!(
        crate::poly::grm_logprobs(1.0, &[1e-18, 0.0])[1],
        f64::NEG_INFINITY
    );
    for specific_count in [0, 1] {
        for nodes in [3, 2] {
            let cfg = TwoTierGrmConfig {
                estimate_primary_correlation: false,
                q_primary: nodes,
                q_specific: 1,
                max_iter: 1,
                tol: 1e-6,
                n_starts: 1,
                seed: 0,
                newton_iter: 1,
                ridge: 1e-8,
            };
            let sm = vec![if specific_count == 0 { -1 } else { 0 }; 2];
            let v = validate_data(
                &[1, 1],
                None,
                &[true, true],
                &sm,
                1,
                2,
                1,
                specific_count,
                3,
                &cfg,
                false,
            )
            .unwrap();
            let (tz, wz) = gh_rule(nodes).unwrap();
            let (ts, ws) = gh_rule(1).unwrap();
            let (coords, lw) = build_primary_grid(tz, wz, 1, v.grid_size);
            let lws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
            let parameters = vec![
                ItemParams {
                    a_p: vec![1.0],
                    a_s: if specific_count == 0 { None } else { Some(1.0) },
                    d: vec![1e-18, 0.0]
                };
                2
            ];
            let length = 1 + specific_count;
            let mut cpu_m = LatentPosteriorMoments {
                mean: vec![0.0; length],
                second: vec![0.0; length],
            };
            let (cpu_ll, _, _) = e_step_with_moments(
                &v,
                &[1, 1],
                None,
                &parameters,
                &lw,
                &lws,
                &coords,
                ts,
                v.grid_size,
                1,
                Some(&mut cpu_m),
                false,
                None,
            );
            let mut gpu_m = LatentPosteriorMoments {
                mean: vec![0.0; length],
                second: vec![0.0; length],
            };
            let result = e_step_gpu_person_moments(
                &v,
                &[1, 1],
                None,
                &parameters,
                &lw,
                &lws,
                &coords,
                ts,
                Some(&mut gpu_m),
                None,
                true,
                1 << 20,
            );
            if nodes == 3 {
                assert!(cpu_ll.is_finite());
                assert_eq!(result.unwrap().to_bits(), cpu_ll.to_bits());
                assert_eq!(gpu_m.mean, cpu_m.mean);
                assert_eq!(gpu_m.second, cpu_m.second);
            } else {
                assert!(!cpu_ll.is_finite());
                assert!(
                    result.is_err(),
                    "all-node zero mass must reject without fallback"
                );
            }
        }
    }
    println!("actual_finite_GRM_zero_probability_node_CPU_GPU_cases=4 passed");

    // 같은 seed·제약·prior·노드에서 수렴 후 문항 모수도 대조한다.
    // 작은 합성 표본의 극단 모수를 likelihood/EAP 근접만으로 수용하지 않는다.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/two_tier_reference_gpu/sparse_p2_s4.json"
    ))
    .unwrap();
    let y: Vec<usize> = serde_json::from_value(fixture["y"].clone()).unwrap();
    let primary: Vec<bool> = serde_json::from_value(fixture["primary_map"].clone()).unwrap();
    let specific: Vec<i32> = serde_json::from_value(fixture["specific_map"].clone()).unwrap();
    let size = |key: &str| fixture[key].as_u64().unwrap() as usize;
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        q_primary: size("q_primary"),
        q_specific: size("q_specific"),
        max_iter: size("max_iter"),
        tol: fixture["tol"].as_f64().unwrap(),
        n_starts: size("n_starts"),
        seed: fixture["seed"].as_u64().unwrap(),
        newton_iter: 10,
        ridge: 1e-8,
    };
    let fit = |device| {
        fit_two_tier_grm_with_device(
            &y,
            None,
            &primary,
            &specific,
            size("n_persons"),
            size("n_items"),
            size("n_primary"),
            size("n_specific"),
            size("n_cat"),
            &cfg,
            device,
            Some(1 << 28),
        )
        .unwrap()
    };
    REFERENCE_TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
    let cpu = fit(Device::Cpu);
    let cpu_trace = REFERENCE_TRACE.with(|trace| trace.borrow_mut().take().unwrap());
    REFERENCE_TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
    let gpu = fit(Device::Gpu);
    let gpu_trace = REFERENCE_TRACE.with(|trace| trace.borrow_mut().take().unwrap());
    let parameter_gap = |a: &[crate::two_tier_grm::ItemParams],
                         b: &[crate::two_tier_grm::ItemParams]| {
        a.iter()
            .zip(b)
            .flat_map(|(a, b)| {
                a.a_p
                    .iter()
                    .chain(a.a_s.iter())
                    .chain(&a.d)
                    .zip(b.a_p.iter().chain(b.a_s.iter()).chain(&b.d))
            })
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };
    assert_eq!(cpu_trace.len(), gpu_trace.len());
    let full_parameter_gap = cpu_trace
        .iter()
        .zip(&gpu_trace)
        .map(|(a, b)| parameter_gap(&a.params, &b.params))
        .fold(0.0, f64::max);
    let full_loglik_gap = cpu_trace
        .iter()
        .zip(&gpu_trace)
        .map(|(a, b)| (a.loglik - b.loglik).abs())
        .fold(0.0, f64::max);
    println!("strict_synthetic_cpu_iterations={}, gpu_iterations={}, full_trajectory_parameter_max_delta={full_parameter_gap}, full_trajectory_loglik_max_delta={full_loglik_gap}",cpu.n_iter,gpu.n_iter);
    let first = cpu_trace
        .iter()
        .zip(&gpu_trace)
        .position(|(a, b)| parameter_gap(&a.params, &b.params) > 1e-3);
    if let Some(first) = first {
        for index in first.saturating_sub(1)..=first {
            let (a, b) = (&cpu_trace[index], &gpu_trace[index]);
            let count_gap = a
                .item13_counts
                .iter()
                .flatten()
                .zip(b.item13_counts.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            println!("first_parity_crossing_iteration={}, parameter_max_delta={}, item13_count_max_delta={}, certified_loglik_delta={}",
                a.iteration, parameter_gap(&a.params,&b.params), count_gap, b.loglik-a.loglik);
        }
    }
    assert!(cpu.converged && gpu.converged);
    assert_eq!(cpu.n_iter, gpu.n_iter);

    // 동일한 CPU 모수 상태를 고정해 table 양자화와 GPU sweep 오차를 분리한다.
    use crate::two_tier_grm::{
        build_primary_grid, chol_inverse, cholesky_lower, e_step_gpu_person_moments,
        e_step_with_moments, gh_rule, item_logprob_tables, item_neg_ll_grad, m_step_item,
        reweighted_log_weights, validate, ReducedFitStatistics,
    };
    let v = validate(
        &y,
        None,
        &primary,
        &specific,
        size("n_persons"),
        size("n_items"),
        size("n_primary"),
        size("n_specific"),
        size("n_cat"),
        &cfg,
    )
    .unwrap();
    let (tz, wz) = gh_rule(cfg.q_primary).unwrap();
    let (ts, ws) = gh_rule(cfg.q_specific).unwrap();
    let (coords, log_w0) = build_primary_grid(tz, wz, v.n_primary, v.grid_size);
    let (chol, logdet) = cholesky_lower(&cpu.phi, v.n_primary).unwrap();
    let log_w = reweighted_log_weights(
        &log_w0,
        &coords,
        &chol_inverse(&chol, v.n_primary),
        logdet,
        v.n_primary,
    );
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let frozen_iteration = first
        .map(|index| index.saturating_sub(1))
        .unwrap_or(102.min(cpu.n_iter));
    let params = cpu_trace[frozen_iteration].params.clone();
    println!("fixed_cpu_state_iteration={frozen_iteration}");
    let tables = item_logprob_tables(&v, &params, &coords, ts, v.grid_size).unwrap();
    let quantized: Vec<Vec<f64>> = tables
        .iter()
        .map(|row| row.iter().map(|&x| f64::from(x as f32)).collect())
        .collect();
    let sweep = |tables: &[Vec<f64>]| {
        e_step_with_moments(
            &v,
            &y,
            None,
            &params,
            &log_w,
            &log_ws,
            &coords,
            ts,
            v.grid_size,
            ts.len(),
            None,
            true,
            Some(tables),
        )
    };
    let (_, exact_counts, _) = sweep(&tables);
    let (_, quantized_counts, _) = sweep(&quantized);
    let mut stats = ReducedFitStatistics {
        counts: Vec::new(),
        cross: Vec::new(),
    };
    e_step_gpu_person_moments(
        &v,
        &y,
        None,
        &params,
        &log_w,
        &log_ws,
        &coords,
        ts,
        None,
        Some(&mut stats),
        false,
        1 << 28,
    )
    .unwrap();
    let mut precise_stats = ReducedFitStatistics {
        counts: Vec::new(),
        cross: Vec::new(),
    };
    e_step_gpu_person_moments(
        &v,
        &y,
        None,
        &params,
        &log_w,
        &log_ws,
        &coords,
        ts,
        None,
        Some(&mut precise_stats),
        true,
        1 << 28,
    )
    .unwrap();
    // 작은 예산에서 3인 batch/마지막 1인 batch를 강제해 고정 자원 재사용을 검증한다.
    let stream_started=std::time::Instant::now();
    let mut streamed_stats=ReducedFitStatistics {counts:Vec::new(),cross:Vec::new()};
    e_step_gpu_person_moments(&v,&y,None,&params,&log_w,&log_ws,&coords,ts,
        None,Some(&mut streamed_stats),true,256*1024).unwrap();
    assert_eq!(streamed_stats.counts,exact_counts);
    println!("fixed_cpu_state_cached_streamed_budget_bytes=262144, expected_person_batch=3_then_short_1, counts_max_delta=0, entire_E_step_seconds={}",stream_started.elapsed().as_secs_f64());
    let i = 13;
    let mut packed: Vec<f64> = v.free_primaries[i]
        .iter()
        .map(|&d| params[i].a_p[d])
        .collect();
    packed.push(params[i].a_s.unwrap());
    packed.extend_from_slice(&params[i].d);
    let grad = |counts: &[Vec<f64>]| {
        item_neg_ll_grad(
            &packed,
            &v.free_primaries[i],
            true,
            &coords,
            ts,
            v.n_primary,
            v.grid_size,
            ts.len(),
            counts,
            v.n_cat,
        )
        .1
    };
    let update = |counts: &[Vec<f64>]| {
        m_step_item(
            packed.clone(),
            &v.free_primaries[i],
            true,
            &coords,
            ts,
            v.n_primary,
            v.grid_size,
            ts.len(),
            counts,
            v.n_cat,
            cfg.ridge,
            cfg.newton_iter,
        )
    };
    let hessian_diagnostic = |counts: &[Vec<f64>]| {
        let gradient = grad(counts);
        let n = packed.len();
        let mut h = vec![0.0; n * n];
        for col in 0..n {
            let mut point = packed.clone();
            point[col] += 1e-5;
            let g = item_neg_ll_grad(
                &point,
                &v.free_primaries[i],
                true,
                &coords,
                ts,
                v.n_primary,
                v.grid_size,
                ts.len(),
                counts,
                v.n_cat,
            )
            .1;
            for row in 0..n {
                h[row * n + col] = (g[row] - gradient[row]) / 1e-5;
            }
        }
        // 원래 solver의 행렬 생성 순서를 재현한 뒤 대칭 부분만 진단한다.
        for row in 0..n {
            for col in 0..n {
                h[row * n + col] = 0.5 * (h[row * n + col] + h[col * n + row]);
            }
            h[row * n + row] += cfg.ridge;
        }
        let original = h.clone();
        for row in 0..n {
            for col in 0..n {
                h[row * n + col] = 0.5 * (original[row * n + col] + original[col * n + row]);
            }
        }
        crate::factor::symmetric_eigen_desc(&h, n).unwrap().0
    };
    let exact_eigenvalues = hessian_diagnostic(&exact_counts[i]);
    println!("fixed_cpu_state_item13_f64_FD_hessian_symmetric_part_eigenvalues_with_original_ridge={exact_eigenvalues:?}");
    let exact_gradient = grad(&exact_counts[i]);
    let exact_update = update(&exact_counts[i]);
    for (name, counts) in [
        ("table_quantization", &quantized_counts),
        ("gpu_sweep", &stats.counts),
        ("gpu_word_products", &precise_stats.counts),
    ] {
        let count_gap = exact_counts
            .iter()
            .flatten()
            .flatten()
            .zip(counts.iter().flatten().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let gradient_gap = exact_gradient
            .iter()
            .zip(grad(&counts[i]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let update_gap = exact_update
            .iter()
            .zip(update(&counts[i]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let eigenvalues = hessian_diagnostic(&counts[i]);
        println!("fixed_cpu_state_{name}: count_max_delta={count_gap}, item13_gradient_max_delta={gradient_gap}, item13_Newton_update_max_delta={update_gap}, FD_hessian_symmetric_part_eigenvalues={eigenvalues:?}");
    }
    let limits = crate::gpu::GpuContext::get().unwrap().device.limits();
    println!(
        "actual_gpu_storage_binding_bytes={}, actual_gpu_buffer_bytes={}",
        limits.max_storage_buffer_binding_size, limits.max_buffer_size
    );
    assert!(
        full_parameter_gap < 1e-3,
        "full parameter trajectory parity failed: {full_parameter_gap}"
    );
    for (name, left, right) in [
        ("a_primary", &cpu.a_primary, &gpu.a_primary),
        ("a_specific", &cpu.a_specific, &gpu.a_specific),
        ("threshold", &cpu.threshold, &gpu.threshold),
    ] {
        let gap = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("strict_synthetic_{name}_max_delta={gap}");
        assert!(
            gap < 1e-3,
            "strict synthetic {name} parameter parity failed: {gap}"
        );
    }
}

#[test]
fn rejects_malformed_primary_map() {
    let (y, n_persons) = tiny_data();
    // Wrong length.
    let short = vec![true; TINY_N_ITEMS * TINY_N_PRIMARY - 1];
    fit_two_tier_grm(
        &y,
        None,
        &short,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("primary_map length must equal n_items * n_primary");
    // A primary with fewer than two loading items is rejected as an
    // implementation stability choice (the primary correlation is weakly
    // identified otherwise; mirrors the stage-1 specific-block rule).
    let mut single = vec![false; TINY_N_ITEMS * TINY_N_PRIMARY];
    for i in 0..TINY_N_ITEMS {
        single[i * TINY_N_PRIMARY] = true; // primary 0: all items
    }
    single[1] = true; // primary 1: one item only (item 0)
    fit_two_tier_grm(
        &y,
        None,
        &single,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("a primary with a single loading item must be rejected");
}

#[test]
fn oracle_rejects_nonzero_slope_at_fixed_primary_positions() {
    // Fixed (pattern-zero) primary positions own no slope parameter: a
    // non-zero value there would be silently dropped, so the oracle fails
    // loudly — the same contract as stage-1's general-only a_specific rule.
    let (_, a_s, thresholds, phi) = tiny_params();
    let (y, n_persons) = tiny_data();
    let (mut a_p, _, _, _) = tiny_params();
    a_p[1] = 0.5; // item 0 does not load primary 1, yet carries a slope
    two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &phi,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("non-zero a_primary at a fixed pattern position must fail loudly");
}

#[test]
fn oracle_rejects_non_correlation_phi() {
    let (a_p, a_s, thresholds, _) = tiny_params();
    let (y, n_persons) = tiny_data();
    // Non-unit diagonal.
    let bad_diag = vec![1.2, 0.35, 0.35, 1.0];
    two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &bad_diag,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("phi with a non-unit diagonal must fail loudly");
    // Symmetric but indefinite (|rho| > 1 forces non-PD).
    let indefinite = vec![1.0, 1.5, 1.5, 1.0];
    two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &indefinite,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("non-positive-definite phi must fail loudly");
}

#[test]
#[cfg_attr(coverage, ignore = "heavy-numeric: slow CPU fit; runs in the non-coverage rust job")]
fn arbitrary_quadrature_counts_above_the_old_fixed_table_are_accepted() {
    // #1929: node count controls integration precision and must not be
    // capped at a fixed table; 5/22/100 used to be rejected, now must fit
    // cleanly (module reuses `quadrature::require_gh_rule`, any n >= 1).
    let (y, n_persons) = tiny_data();
    for (qp, qs) in [(5, 7), (7, 5), (15, 22), (100, 7)] {
        let cfg = TwoTierGrmConfig {
            q_primary: qp,
            q_specific: qs,
            ..valid_config()
        };
        fit_two_tier_grm(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("q_primary={qp}, q_specific={qs} must be accepted: {e}"));
    }
}

#[test]
fn rejects_zero_quadrature_counts() {
    let (y, n_persons) = tiny_data();
    for (qp, qs) in [(0, 7), (7, 0)] {
        let cfg = TwoTierGrmConfig {
            q_primary: qp,
            q_specific: qs,
            ..valid_config()
        };
        let err = fit_two_tier_grm(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("q_primary/q_specific == 0 must fail loudly, never clamp");
        assert!(
            err.contains("q_primary") || err.contains("q_specific") || err.contains('q'),
            "quadrature error must name the offending argument; got: {err}"
        );
    }
}

#[test]
fn identity_rejects_identical_primary_support_but_accepts_nested_support() {
    let (y, n_persons) = tiny_data();
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        max_iter: 500,
        ..valid_config()
    };
    let shared = [true; TINY_N_ITEMS * TINY_N_PRIMARY];
    let err = fit_two_tier_grm(
        &y,
        None,
        &shared,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect_err("identical supports admit a continuous rotation");
    assert!(err.contains("identical free-loading item sets"), "{err}");

    let nested: Vec<bool> = (0..TINY_N_ITEMS).flat_map(|i| [true, i < 5]).collect();
    let fit = fit_two_tier_grm(
        &y,
        None,
        &nested,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("distinct nested supports must fit");
    assert!(fit.converged, "nested-support fit did not converge");
    assert_ne!(fit.termination_reason, "max_iter_reached");
    assert_eq!(fit.phi, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn rejects_bad_iteration_and_tolerance_budgets() {
    let (y, n_persons) = tiny_data();
    for cfg in [
        TwoTierGrmConfig {
            max_iter: 0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            tol: 0.0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            tol: f64::NAN,
            ..valid_config()
        },
        TwoTierGrmConfig {
            n_starts: 0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            newton_iter: 0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            ridge: -1e-8,
            ..valid_config()
        },
    ] {
        fit_two_tier_grm(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("out-of-range caller budgets must fail loudly, never clamp");
    }
}

#[test]
fn unobserved_category_fails_loudly_with_item_and_category() {
    let n_persons = 30usize;
    let mut y = vec![0usize; n_persons * TINY_N_ITEMS];
    for p in 0..n_persons {
        for i in 0..TINY_N_ITEMS {
            y[p * TINY_N_ITEMS + i] = p % TINY_N_CAT;
        }
        y[p * TINY_N_ITEMS] = p % 2; // item 0: only categories 0-1
    }
    let err = fit_two_tier_grm(
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("an unobserved category must fail loudly, never be imputed");
    assert!(
        err.contains("item 0") && err.contains("category 2"),
        "the error must name the item and the unobserved category; got: {err}"
    );
}

#[test]
fn non_convergence_is_reported_not_substituted() {
    let (y, _) = tiny_data();
    let n_big = 200usize;
    let mut y_big = Vec::with_capacity(n_big * TINY_N_ITEMS);
    for p in 0..n_big {
        for i in 0..TINY_N_ITEMS {
            y_big.push(y[(p % 12) * TINY_N_ITEMS + i]);
        }
    }
    let cfg = TwoTierGrmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..valid_config()
    };
    let fit = fit_two_tier_grm(
        &y_big,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_big,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("max_iter exhaustion must be reported via flags, not via Err");
    assert!(
        !fit.converged,
        "a single EM sweep must not report convergence"
    );
    assert_eq!(
        fit.termination_reason, "max_iter_reached",
        "termination reason must say max_iter_reached"
    );
}

// Same-node oracle for Cai (2010), pp. 608-609, Appendices A/B. Enumerating
// every specific-node combination independently checks the reduced posterior
// without claiming continuous-integral accuracy or empirical fit recovery.
#[test]
fn latent_moments_match_full_grid_with_missing_blocks() {
    use crate::two_tier_grm::{
        e_step, e_step_with_moments, ItemParams, LatentPosteriorMoments, Validated,
    };
    let v = Validated {
        n_persons: 3,
        n_items: 5,
        n_primary: 2,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: 4,
        free_primaries: vec![vec![0, 1]; 5],
        blocks: vec![vec![0, 1], vec![2, 3]],
        specific_free: vec![4],
        item_block: vec![Some(0), Some(0), Some(1), Some(1), None],
    };
    let pars: Vec<ItemParams> = (0..5)
        .map(|i| ItemParams {
            a_p: vec![0.7 + i as f64 * 0.1, -0.4 + i as f64 * 0.2],
            a_s: if i < 4 {
                Some(0.5 - i as f64 * 0.3)
            } else {
                None
            },
            d: vec![0.8, -0.6],
        })
        .collect();
    // Asymmetric, non-centered nodes/weights exercise first moments.
    let coords = [-0.5, 0.2, -0.5, 1.1, 0.9, 0.2, 0.9, 1.1];
    let wg: [f64; 4] = [0.1, 0.2, 0.3, 0.4];
    let ts: [f64; 2] = [-0.7, 1.2];
    let ws: [f64; 2] = [0.6, 0.4];
    let log_w: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0];
    let obs = [
        true, true, true, true, true, false, false, true, true, true, false, false, false, false,
        false,
    ];
    let mut moments = LatentPosteriorMoments {
        mean: vec![123.0; 12],
        second: vec![456.0; 12],
    };
    let result = e_step_with_moments(
        &v,
        &y,
        Some(&obs),
        &pars,
        &log_w,
        &log_ws,
        &coords,
        &ts,
        4,
        2,
        Some(&mut moments),
        true,
        None,
    );
    let old = e_step(
        &v,
        &y,
        Some(&obs),
        &pars,
        &log_w,
        &log_ws,
        &coords,
        &ts,
        4,
        2,
    );
    assert_eq!(
        result, old,
        "opting into moments must preserve existing outputs"
    );
    let expected_mean = moments.mean.clone();
    let expected_second = moments.second.clone();
    let no_counts = e_step_with_moments(
        &v,
        &y,
        Some(&obs),
        &pars,
        &log_w,
        &log_ws,
        &coords,
        &ts,
        4,
        2,
        Some(&mut moments),
        false,
        None,
    );
    assert!(no_counts.1.is_empty());
    assert_eq!(no_counts.0, result.0);
    assert_eq!(no_counts.2, result.2);
    assert_eq!(moments.mean, expected_mean);
    assert_eq!(moments.second, expected_second);

    let mut brute_ll = 0.0;
    for person in 0..3 {
        let mut mass = 0.0;
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        for g in 0..4 {
            for h0 in 0..2 {
                for h1 in 0..2 {
                    let latent = [coords[g * 2], coords[g * 2 + 1], ts[h0], ts[h1]];
                    let mut weight = wg[g] * ws[h0] * ws[h1];
                    for i in 0..5 {
                        if !obs[person * 5 + i] {
                            continue;
                        }
                        let mut base = pars[i].a_p[0] * latent[0] + pars[i].a_p[1] * latent[1];
                        if let Some(s) = v.item_block[i] {
                            base += pars[i].a_s.unwrap() * latent[2 + s];
                        }
                        // Full-grid oracle uses probabilities, not reduced log-posterior algebra.
                        weight *=
                            crate::poly::grm_logprobs(base, &pars[i].d)[y[person * 5 + i]].exp();
                    }
                    mass += weight;
                    for d in 0..4 {
                        first[d] += weight * latent[d];
                        second[d] += weight * latent[d] * latent[d];
                    }
                }
            }
        }
        brute_ll += mass.ln();
        for d in 0..4 {
            assert!(
                (moments.mean[person * 4 + d] - first[d] / mass).abs() < 1e-12,
                "first moment person {person}, dimension {d}"
            );
            assert!(
                (moments.second[person * 4 + d] - second[d] / mass).abs() < 1e-12,
                "second moment person {person}, dimension {d}"
            );
        }
    }
    assert!((result.0 - brute_ll).abs() < 1e-12);
    // Entirely missing persons retain the finite quadrature prior.
    for s in 0..2 {
        assert!((moments.mean[8 + 2 + s] - (ws[0] * ts[0] + ws[1] * ts[1])).abs() < 1e-12);
    }
}

#[test]
fn focal_scores_match_physical_full_grid_and_missing_prior() {
    use crate::two_tier_grm::{gh_rule, score_two_tier_grm_orthogonal};
    let a_p = [0.7, -0.4, 0.8, -0.2, 0.9, 0.1, 1.0, 0.2, 1.1, 0.4];
    let a_s = [0.5, 0.2, -0.1, -0.4, 0.0];
    let threshold = [0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6];
    let primary = [true; 10];
    let specific = [0, 0, 1, 1, -1];
    let mean = [0.4, -0.2, 0.7, -0.6];
    let sd = [1.1, 0.8, 0.9, 1.3];
    let y = [0, 1, 2, 0, 1, 0, 0, 0, 0, 0];
    let observed = [
        true, true, true, true, true, false, false, false, false, false,
    ];
    let bank_before = (a_p, a_s, threshold);
    let score = score_two_tier_grm_orthogonal(
        &a_p,
        &a_s,
        &threshold,
        &mean,
        &sd,
        &y,
        Some(&observed),
        &primary,
        &specific,
        2,
        5,
        2,
        2,
        3,
        7,
        7,
    )
    .unwrap();
    assert_eq!((a_p, a_s, threshold), bank_before);
    let (z, w) = gh_rule(7).unwrap();
    let mut full_loglik = 0.0;
    for person in 0..2 {
        let mut mass = 0.0;
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        // Physical latent grid keeps original slopes/intercepts untouched.
        for g0 in 0..7 {
            for g1 in 0..7 {
                for h0 in 0..7 {
                    for h1 in 0..7 {
                        let index = [g0, g1, h0, h1];
                        let mut latent = [0.0; 4];
                        let mut weight = 1.0;
                        for d in 0..4 {
                            latent[d] = mean[d] + sd[d] * z[index[d]];
                            weight *= w[index[d]];
                        }
                        for i in 0..5 {
                            if !observed[person * 5 + i] {
                                continue;
                            }
                            let mut base = a_p[i * 2] * latent[0] + a_p[i * 2 + 1] * latent[1];
                            if specific[i] >= 0 {
                                base += a_s[i] * latent[2 + specific[i] as usize];
                            }
                            weight *= crate::poly::grm_logprobs(base, &threshold[i * 2..i * 2 + 2])
                                [y[person * 5 + i]]
                                .exp();
                        }
                        mass += weight;
                        for d in 0..4 {
                            first[d] += weight * latent[d];
                            second[d] += weight * latent[d] * latent[d];
                        }
                    }
                }
            }
        }
        full_loglik += mass.ln();
        for d in 0..4 {
            assert!((score.mean[person * 4 + d] - first[d] / mass).abs() < 1e-12);
            assert!((score.second[person * 4 + d] - second[d] / mass).abs() < 1e-12);
            let var = second[d] / mass - (first[d] / mass).powi(2);
            assert!((score.sd[person * 4 + d] - var.sqrt()).abs() < 1e-12);
        }
    }
    assert!((score.loglik - full_loglik).abs() < 1e-12);
    for d in 0..4 {
        assert!((score.mean[4 + d] - mean[d]).abs() < 1e-12);
        assert!((score.sd[4 + d] - sd[d]).abs() < 1e-12);
    }
    let call = |mu: &[f64], sigma: &[f64], observed: &[bool], responses: &[usize]| {
        score_two_tier_grm_orthogonal(
            &a_p,
            &a_s,
            &threshold,
            mu,
            sigma,
            responses,
            Some(observed),
            &primary,
            &specific,
            2,
            5,
            2,
            2,
            3,
            7,
            7,
        )
    };
    assert!(call(&mean, &[1.0, 0.0, 1.0, 1.0], &observed, &y).is_err());
    assert!(call(&mean, &[1.0, f64::NAN, 1.0, 1.0], &observed, &y).is_err());
    assert!(call(&[f64::INFINITY, 0.0, 0.0, 0.0], &sd, &observed, &y).is_err());
    assert!(call(&[0.0], &sd, &observed, &y).is_err());
    let mut bad = y;
    bad[0] = 3;
    assert!(call(&mean, &sd, &observed, &bad).is_err());
    // Missing placeholders are ignored when their observation flag is false.
    bad[0] = 0;
    bad[5] = usize::MAX;
    assert!(call(&mean, &sd, &observed, &bad).is_ok());
    assert!(call(&[1e308; 4], &sd, &observed, &y).is_err());
    // The same fixed-bank responses may omit categories for scoring, while
    // the existing fitter still requires category coverage before estimating.
    let cfg = crate::two_tier_grm::TwoTierGrmConfig {
        estimate_primary_correlation: true,
        q_primary: 7,
        q_specific: 7,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
    };
    let fitted_validation = crate::two_tier_grm::validate(
        &y,
        Some(&observed),
        &primary,
        &specific,
        2,
        5,
        2,
        2,
        3,
        &cfg,
    );
    assert!(matches!(fitted_validation, Err(message) if message.contains("never observed")));
}

#[test]
fn focal_em_preserves_one_step_moments_and_termination_receipts() {
    use crate::two_tier_grm::{fit_two_tier_grm_focal_orthogonal, score_two_tier_grm_orthogonal};
    let ap = [0.7, -0.4, 0.8, -0.2, 0.9, 0.1, 1.0, 0.2, 1.1, 0.4];
    let asp = [0.5, 0.2, -0.1, -0.4, 0.0];
    let threshold = [0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6];
    let primary = [true; 10];
    let specific = [0, 0, 1, 1, -1];
    let y = [0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 2, 0];
    let mu = [0.4, -0.2, 0.7, -0.6];
    let sd = [1.1, 0.8, 0.9, 1.3];
    let initial = score_two_tier_grm_orthogonal(
        &ap, &asp, &threshold, &mu, &sd, &y, None, &primary, &specific, 3, 5, 2, 2, 3, 7, 7,
    )
    .unwrap();
    let run = |q, cap, tol, observed: Option<&[bool]>| {
        fit_two_tier_grm_focal_orthogonal(
            &ap, &asp, &threshold, &mu, &sd, &y, observed, &primary, &specific, 3, 5, 2, 2, 3, q,
            q, cap, tol,
        )
    };
    let fit = run(7, 1, 1e-14, None).unwrap();
    assert_eq!(fit.n_iter, 1);
    assert_eq!(fit.loglik_trace.len(), 2);
    assert_eq!(fit.loglik_trace[0], initial.loglik);
    assert_eq!(fit.loglik_trace[1], fit.scores.loglik);
    assert_eq!(
        fit.final_loglik_change,
        fit.loglik_trace[1] - fit.loglik_trace[0]
    );
    assert_eq!(fit.initial_mean, mu);
    assert_eq!(fit.initial_sd, sd);
    assert_eq!(fit.q_primary, 7);
    assert_eq!(fit.q_specific, 7);
    assert_eq!(fit.max_iter, 1);
    assert_eq!(fit.tol, 1e-14);
    for d in 0..4 {
        let m = (0..3)
            .map(|person| initial.mean[person * 4 + d])
            .sum::<f64>()
            / 3.0;
        // Independent raw-second-moment identity at moderate mean values.
        let var = (0..3)
            .map(|person| initial.second[person * 4 + d])
            .sum::<f64>()
            / 3.0
            - m * m;
        assert!((fit.latent_mean[d] - m).abs() < 1e-12);
        assert!((fit.latent_sd[d] - var.sqrt()).abs() < 1e-12);
    }
    for q in [2, 7] {
        let result = run(q, 12, 1e-6, None).unwrap();
        assert_eq!(result.loglik_trace.len(), result.n_iter + 1);
        assert_eq!(*result.loglik_trace.last().unwrap(), result.scores.loglik);
        assert_eq!(
            result.converged,
            result.termination_reason == "tolerance_met"
        );
        match result.termination_reason {
            "loglik_decreased" => assert!(result.final_loglik_change < 0.0),
            "tolerance_met" => assert!(
                result.final_loglik_change >= 0.0 && result.final_loglik_change <= result.tol
            ),
            "max_iter_reached" => assert_eq!(result.n_iter, result.max_iter),
            other => panic!("unexpected termination {other}"),
        }
        eprintln!(
            "toy q={q}: {}, updates={}",
            result.termination_reason, result.n_iter
        );
    }
    // Moving coarse GH nodes can decrease the evaluated likelihood. Retain
    // that evaluated state and stop without reporting convergence.
    let decreased = fit_two_tier_grm_focal_orthogonal(
        &ap, &asp, &threshold, &[0.0; 4], &[2.0; 4], &y, None, &primary, &specific, 3, 5, 2, 2, 3,
        2, 2, 30, 1e-12,
    )
    .unwrap();
    assert!(!decreased.converged);
    assert_eq!(decreased.termination_reason, "loglik_decreased");
    assert!(decreased.final_loglik_change < 0.0);
    assert!(decreased.n_iter < decreased.max_iter);
    assert_eq!(
        *decreased.loglik_trace.last().unwrap(),
        decreased.scores.loglik
    );
    // Loose synthetic tolerance tests the convergence branch, not a study setting.
    let stopped = run(7, 1, 1e6, None).unwrap();
    assert!(stopped.converged);
    assert_eq!(stopped.termination_reason, "tolerance_met");
    assert_eq!(stopped.n_iter, 1);
    assert!(run(7, 0, 1e-6, None).is_err());
    assert!(run(7, 1, 0.0, None).is_err());
    assert!(run(7, 1, f64::INFINITY, None).is_err());
    assert!(run(7, 1, 1e-6, Some(&[false; 15])).is_err());
    let mut mask = [true; 15];
    for person in 0..3 {
        mask[person * 5 + 2] = false;
        mask[person * 5 + 3] = false;
    }
    assert!(run(7, 1, 1e-6, Some(&mask)).is_err());
}

/// Known-population diagnostic for the fixed-item Gaussian EM application.
/// Source: Cai (2010), DOI 10.1007/s11336-010-9178-0, pp. 608-609,
/// Appendices A/B (Gaussian latent-density objective and posterior moments).
/// The fixture independently integrates response-pattern probabilities over
/// two physical latent coordinates. Rounded expected counts remove RNG noise.
/// This bank, count, quadrature and error tolerance are test choices; the
/// source does not prescribe them or establish study-model identification.
#[test]
#[cfg_attr(coverage, ignore = "heavy-numeric: slow CPU fit; runs in the non-coverage rust job")]
fn focal_gaussian_recovers_declared_distribution() {
    use crate::two_tier_grm::fit_two_tier_grm_focal_orthogonal;
    let ap = [1.2, 1.6, 0.5, 0.8];
    let asp = [0.0, 0.0, 1.4, -1.6];
    let d = [0.8, -0.7, 0.4, -1.0, 1.0, -0.5, 0.6, -0.9];
    let pm = [true; 4];
    let sm = [-1, -1, 0, 0];
    let truth_mean = [0.4, -0.3];
    let truth_sd = [1.2, 0.8];
    let (nodes, weights) = crate::quadrature::gh_rule(31).unwrap();
    let mut probabilities = vec![0.0_f64; 81];
    for (g, &z) in nodes.iter().enumerate() {
        let primary = truth_mean[0] + truth_sd[0] * z;
        for (h, &zs) in nodes.iter().enumerate() {
            let specific = truth_mean[1] + truth_sd[1] * zs;
            let mut item = [[0.0; 3]; 4];
            for i in 0..4 {
                let eta = ap[i] * primary + asp[i] * specific;
                let upper = 1.0 / (1.0 + (-(eta + d[2 * i])).exp());
                let lower = 1.0 / (1.0 + (-(eta + d[2 * i + 1])).exp());
                item[i] = [1.0 - upper, upper - lower, lower];
            }
            for (pattern, probability) in probabilities.iter_mut().enumerate() {
                let mut code = pattern;
                let mut value = weights[g] * weights[h];
                for categories in &item {
                    value *= categories[code % 3];
                    code /= 3;
                }
                *probability += value;
            }
        }
    }
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let mut y = Vec::new();
    for (pattern, probability) in probabilities.iter().enumerate() {
        let count = (4000.0 * probability).round() as usize;
        let mut code = pattern;
        let mut row = [0; 4];
        for category in &mut row {
            *category = code % 3;
            code /= 3;
        }
        for _ in 0..count {
            y.extend_from_slice(&row);
        }
    }
    let n = y.len() / 4;
    for q in [15, 21] {
        let fit = fit_two_tier_grm_focal_orthogonal(
            &ap, &asp, &d, &[0.0; 2], &[1.0; 2], &y, None, &pm, &sm, n, 4, 1, 1, 3, q, q, 300, 1e-6,
        )
        .unwrap();
        eprintln!(
            "recovery n={n} q={q} reason={} updates={} mean={:?} sd={:?} delta={}",
            fit.termination_reason,
            fit.n_iter,
            fit.latent_mean,
            fit.latent_sd,
            fit.final_loglik_change
        );
        assert!(fit.converged, "{}", fit.termination_reason);
        for dim in 0..2 {
            assert!((fit.latent_mean[dim] - truth_mean[dim]).abs() < 0.04);
            assert!((fit.latent_sd[dim] - truth_sd[dim]).abs() < 0.04);
        }
    }
}

/// Same-node posterior oracle for shared G/W and four specific blocks.
/// Cai (2010), DOI 10.1007/s11336-010-9178-0, pp.589-590 eqs.11-12,15-16 and
/// pp.608-609 Appendices A/B: integrate shared primary dimensions jointly,
/// and use posterior first/second moments for EAP and variance.
/// This synthetic 16-item layout mirrors the study's G+4+W loading pattern;
/// parameters and five-point nodes are test choices, not study estimates or
/// evidence of quadrature adequacy, identification or population recovery.
#[test]
fn six_latent_crossed_primary_scores_match_full_product() {
    use crate::two_tier_grm::score_two_tier_grm_orthogonal;
    let wording = [4_usize, 5, 6, 9, 12, 14, 15];
    let mut ap = vec![0.0; 32];
    let mut asp = vec![0.0; 16];
    let mut threshold = vec![0.0; 48];
    let mut pm = vec![false; 32];
    let sm: Vec<i32> = (0..16).map(|i| (i / 4) as i32).collect();
    for i in 0..16 {
        pm[2 * i] = true;
        ap[2 * i] = 0.65 + 0.15 * (i % 4) as f64;
        if wording.contains(&i) {
            pm[2 * i + 1] = true;
            ap[2 * i + 1] = if i % 2 == 0 { 0.45 } else { -0.5 };
        }
        asp[i] = 0.35 + 0.2 * (i % 3) as f64;
        for (k, value) in [1.0, 0.1, -0.9].iter().enumerate() {
            threshold[3 * i + k] = value + 0.04 * (i % 5) as f64;
        }
    }
    let mu = [0.4, -0.3, 0.2, -0.5, 0.6, -0.2];
    let sd = [1.2, 0.8, 0.9, 1.1, 0.7, 1.3];
    let y: Vec<usize> = (0..48).map(|i| (i + i / 16) % 4).collect();
    let mut mask = vec![true; 48];
    for i in 0..16 {
        mask[16 + i] = !(4..8).contains(&i) && i != 12;
        mask[32 + i] = false;
    }
    let scores = score_two_tier_grm_orthogonal(
        &ap,
        &asp,
        &threshold,
        &mu,
        &sd,
        &y,
        Some(&mask),
        &pm,
        &sm,
        3,
        16,
        2,
        4,
        4,
        5,
        5,
    )
    .unwrap();
    let (nodes, weights) = crate::quadrature::gh_rule(5).unwrap();
    let mut total_loglik = 0.0;
    for person in 0..3 {
        let mut mass = 0.0;
        let mut first = [0.0; 6];
        let mut second = [0.0; 6];
        for code in 0..5_usize.pow(6) {
            let mut remaining = code;
            let mut theta = [0.0; 6];
            let mut probability = 1.0;
            for dim in 0..6 {
                let h = remaining % 5;
                remaining /= 5;
                theta[dim] = mu[dim] + sd[dim] * nodes[h];
                probability *= weights[h];
            }
            for i in 0..16 {
                if !mask[person * 16 + i] {
                    continue;
                }
                let eta = ap[2 * i] * theta[0]
                    + ap[2 * i + 1] * theta[1]
                    + asp[i] * theta[2 + sm[i] as usize];
                let mut cumulative = [1.0, 0.0, 0.0, 0.0, 0.0];
                for k in 0..3 {
                    cumulative[k + 1] = 1.0 / (1.0 + (-(eta + threshold[3 * i + k])).exp());
                }
                let category = y[person * 16 + i];
                probability *= cumulative[category] - cumulative[category + 1];
            }
            mass += probability;
            for dim in 0..6 {
                first[dim] += probability * theta[dim];
                second[dim] += probability * theta[dim] * theta[dim];
            }
        }
        total_loglik += mass.ln();
        for dim in 0..6 {
            let mean = first[dim] / mass;
            let raw_second = second[dim] / mass;
            let posterior_sd = (raw_second - mean * mean).sqrt();
            assert!((scores.mean[person * 6 + dim] - mean).abs() < 1e-10);
            assert!((scores.second[person * 6 + dim] - raw_second).abs() < 1e-10);
            assert!((scores.sd[person * 6 + dim] - posterior_sd).abs() < 1e-10);
            if person == 2 {
                assert!((scores.mean[person * 6 + dim] - mu[dim]).abs() < 1e-12);
                assert!((scores.sd[person * 6 + dim] - sd[dim]).abs() < 1e-12);
            }
        }
    }
    assert!((scores.loglik - total_loglik).abs() < 1e-10);
    // The six-dimensional crossed-primary bank must pass the mean-rank
    // guard; this is a one-update contract check, not population recovery.
    assert!(crate::two_tier_grm::fit_two_tier_grm_focal_orthogonal(
        &ap,
        &asp,
        &threshold,
        &mu,
        &sd,
        &y,
        Some(&mask),
        &pm,
        &sm,
        3,
        16,
        2,
        4,
        4,
        5,
        5,
        1,
        1e-6,
    )
    .is_ok());
    // Reproduce the installed-native fixture with cargo test -- --nocapture.
    eprintln!("SIX_FIXTURE={{\"a_primary\":{:?},\"a_specific\":{:?},\"threshold\":{:?},\"primary_map\":{:?},\"specific_map\":{:?},\"latent_mean\":{:?},\"latent_sd\":{:?},\"responses\":{:?},\"observed\":{:?},\"person_mean\":{:?},\"person_second\":{:?},\"person_sd\":{:?},\"loglik\":{}}}", ap,asp,threshold,pm,sm,mu,sd,y,mask,scores.mean,scores.second,scores.sd,scores.loglik);
}

/// Free latent means require an injective fixed-item predictor map.
/// Derived from Cai (2010), p.589 eq.11, DOI 10.1007/s11336-010-9178-0:
/// A*v=0 makes mu and mu+v observationally equivalent. The shared pivot
/// guard follows the opened LAPACK DGETF2 Purpose/INFO/pivot loop; its
/// caller threshold is an implementation guard, not DGELSY effective rank.
#[test]
fn focal_mean_rank_rejects_nullspaces_and_preserves_declared_scoring() {
    use crate::two_tier_grm::{fit_two_tier_grm_focal_orthogonal, score_two_tier_grm_orthogonal};
    let ap = [0.8, 1.2, 1.0, 0.6];
    let asp = ap;
    let d = [0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6];
    let y = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2, 1, 0];
    let pm = [true; 4];
    let sm = [0; 4];
    let score = |mu: &[f64]| {
        score_two_tier_grm_orthogonal(
            &ap, &asp, &d, mu, &[1.0; 2], &y, None, &pm, &sm, 4, 4, 1, 1, 3, 5, 5,
        )
        .unwrap()
    };
    // Declared-prior scoring remains meaningful; fitting both means does not.
    let a = score(&[0.0, 0.0]);
    let b = score(&[0.5, -0.5]);
    assert_eq!(a.loglik, b.loglik);
    let fit = fit_two_tier_grm_focal_orthogonal(
        &ap, &asp, &d, &[0.0; 2], &[1.0; 2], &y, None, &pm, &sm, 4, 4, 1, 1, 3, 5, 5, 1, 1e6,
    );
    assert!(matches!(fit, Err(message) if message.contains("column rank")));
    // A third column can be a sum of two independent columns; detecting
    // only pairwise duplicates or nonzero columns would miss this nullspace.
    let ap3 = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    let asp3 = [1.0, 1.0, 2.0, 1.0];
    let pm3 = [true; 8];
    let fit3 = |asp: &[f64], mask: Option<&[bool]>| {
        fit_two_tier_grm_focal_orthogonal(
            &ap3, asp, &d, &[0.0; 3], &[1.0; 3], &y, mask, &pm3, &sm, 4, 4, 2, 1, 3, 5, 5, 1, 1e6,
        )
    };
    assert!(matches!(fit3(&asp3,None),Err(message) if message.contains("column rank")));
    let independent = [0.0, 0.0, 1.0, 1.0];
    assert!(fit3(&independent, None).is_ok());
    let mask: Vec<bool> = (0..16).map(|i| i % 4 == 0 || i % 4 == 2).collect();
    assert!(
        matches!(fit3(&independent,Some(&mask)),Err(message) if message.contains("column rank"))
    );
    // Existing square-Gram users keep their caller thresholds; the direct
    // rectangular path avoids squaring the condition number.
    let rank = |mut matrix: Vec<Vec<f64>>, cols, threshold| {
        crate::lltm::gram_full_rank(&mut matrix, cols, threshold)
    };
    assert!(rank(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 2, 1e-9));
    assert!(!rank(vec![vec![1.0, 1.0], vec![1.0, 1.0]], 2, 1e-9));
    assert!(rank(
        vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, -1.0]],
        2,
        3.0 * f64::EPSILON
    ));
    assert!(rank(
        vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-10]],
        2,
        2.0 * f64::EPSILON
    ));
    assert!(rank(
        vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![0.0, 1.0]],
        2,
        3.0 * f64::EPSILON
    ));
    assert!(!rank(vec![vec![1.0, 0.0]], 2, f64::EPSILON));
    assert!(!rank(vec![vec![1.0], vec![0.0, 1.0]], 2, f64::EPSILON));
    assert!(!rank(vec![vec![f64::NAN]], 1, f64::EPSILON));
    assert!(!rank(vec![vec![1.0]], 1, f64::NAN));
    assert!(!rank(vec![vec![0.0]], 1, 0.0));
}
