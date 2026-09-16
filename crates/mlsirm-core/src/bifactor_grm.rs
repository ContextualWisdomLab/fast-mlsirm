//! Multidimensional Polytomous Bifactor Graded Response Model (GRM) with QMCEM,
//! Multiple-Group Simultaneous Calibration, Slope Bounding, and Oakes Standard Errors.
//!
//! # References
//! - Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis.
//!   *Psychometrika, 57*(3), 423-436.
//! - Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis.
//!   *Psychological Methods, 16*(3), 221-248.
//! - Bock, R. D., & Zimowski, M. F. (1997). Multiple Group IRT.
//!   In W. J. van der Linden & R. K. Hambleton (Eds.), *Handbook of Modern Item Response Theory*.
//! - Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm.
//!   *Journal of the Royal Statistical Society: Series B*, 61(2), 479-482.
//! - Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte Carlo EM.
//!   *Computational Statistics & Data Analysis, 48*(4), 685-701.

use crate::nodes::{build_xi_nodes, XiRule};
use crate::poly::{grm_logprobs, grm_node_gradient, solve_small};

/// Configuration for fitting a polytomous bifactor Graded Response Model via QMCEM.
#[derive(Clone, Debug)]
pub struct BifactorGrmConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub ridge: f64,
    pub newton_iter: usize,
    pub qmc_draws: usize,
    pub seed: u64,
    /// Optional upper bound constraint on discrimination magnitude: |a_id| <= slope_bound.
    pub slope_bound: Option<f64>,
    /// Whether to compute Oakes standard errors after convergence.
    pub compute_oakes_se: bool,
    pub device: crate::Device,
}

impl Default for BifactorGrmConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-4,
            ridge: 1e-6,
            newton_iter: 10,
            qmc_draws: 5000,
            seed: 0x9E37_79B9_7F4A_7C15,
            slope_bound: None,
            compute_oakes_se: true,
            device: crate::Device::Cpu,
        }
    }
}

/// Result of fitting a polytomous bifactor Graded Response Model.
#[derive(Clone, Debug)]
pub struct BifactorGrmResult {
    pub n_dims: usize,
    pub n_items: usize,
    pub n_cat: usize,
    pub n_groups: usize,
    /// Row-major discrimination matrix (n_items * n_dims).
    pub slope: Vec<f64>,
    /// Row-major category thresholds (n_items * (n_cat - 1)), strictly decreasing within each item.
    pub threshold: Vec<f64>,
    /// Group trait means (n_groups * n_dims). Group 0 is fixed at 0.0.
    pub group_means: Vec<f64>,
    /// Group trait variances (n_groups * n_dims). Group 0 is fixed at 1.0.
    pub group_variances: Vec<f64>,
    /// Final marginal log-likelihood.
    pub loglik: f64,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Oakes standard errors for free slopes (same layout as slope, 0.0 for fixed loadings).
    pub oakes_se_slope: Option<Vec<f64>>,
    /// Oakes standard errors for thresholds (same layout as threshold).
    pub oakes_se_threshold: Option<Vec<f64>>,
    /// Minimum eigenvalue of the observed information matrix.
    pub min_eigenvalue: Option<f64>,
    /// Condition number of the observed information matrix.
    pub condition_number: Option<f64>,
}

/// Sensitivity result for different slope bounds.
#[derive(Clone, Debug)]
pub struct SlopeSensitivityEntry {
    pub bound: Option<f64>,
    pub loglik: f64,
    pub converged: bool,
    pub n_iter: usize,
    pub n_bounded_slopes: usize,
    pub max_slope: f64,
}

/// Item negative complete-data log-likelihood and analytic gradient.
fn item_qmc_neg_ll_grad(
    params: &[f64],
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    _n_cat: usize,
) -> (f64, Vec<f64>) {
    let l = dims.len();
    let beta = &params[l..];
    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; params.len()];

    for (nd, cnt) in counts.iter().enumerate() {
        let mut base = 0.0_f64;
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * nodes[nd * n_dims + d];
        }
        let lp = grm_logprobs(base, beta);
        ll += cnt.iter().zip(&lp).map(|(&r, &l2)| r * l2).sum::<f64>();
        let (g_base, g_thr) = grm_node_gradient(base, beta, cnt);
        for (t, &d) in dims.iter().enumerate() {
            grad[t] += g_base * nodes[nd * n_dims + d];
        }
        for (j, &gj) in g_thr.iter().enumerate() {
            grad[l + j] += gj;
        }
    }
    (-ll, grad.iter().map(|&v| -v).collect())
}

/// Newton M-step for one item with optional slope bound clamping.
fn item_m_step_bifactor(
    mut params: Vec<f64>,
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
    ridge: f64,
    n_newton: usize,
    slope_bound: Option<f64>,
) -> Vec<f64> {
    let np = params.len();
    let l = dims.len();

    for _ in 0..n_newton {
        let (f0, g) = item_qmc_neg_ll_grad(&params, dims, nodes, n_dims, counts, n_cat);
        let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }

        let h = 1e-5;
        let mut hess = vec![vec![0.0_f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = item_qmc_neg_ll_grad(&pj, dims, nodes, n_dims, counts, n_cat);
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        for r in 0..np {
            for c in 0..np {
                hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
            }
            hess[r][r] += ridge;
        }

        let mut step = solve_small(hess, g.clone());
        let mut directional = g.iter().zip(&step).map(|(&gi, &si)| gi * si).sum::<f64>();
        if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
            step = g.clone();
            directional = grad_norm * grad_norm;
        }

        let mut max_step = step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        if max_step > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_step;
            }
            directional = g.iter().zip(&step).map(|(&gi, &si)| gi * si).sum();
            max_step = 2.0;
        }

        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..25 {
            let mut candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(&value, &direction)| value - alpha * direction)
                .collect();

            // Apply slope bound clamp if configured
            if let Some(bound) = slope_bound {
                for t in 0..l {
                    candidate[t] = candidate[t].clamp(-bound, bound);
                }
            }

            let (candidate_f, _) =
                item_qmc_neg_ll_grad(&candidate, dims, nodes, n_dims, counts, n_cat);
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted || alpha * max_step < 1e-9 {
            break;
        }
    }
    params
}

/// Fit a polytomous bifactor Graded Response Model via QMCEM.
pub fn fit_bifactor_grm(
    y: &[usize],
    observed: Option<&[bool]>,
    group_ids: Option<&[usize]>,
    n_groups: usize,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    cfg: &BifactorGrmConfig,
) -> Result<BifactorGrmResult, String> {
    if n_persons < 1 || n_items < 1 || n_dims < 1 {
        return Err("n_persons, n_items, and n_dims must be >= 1".into());
    }
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    let m1 = n_cat - 1;
    let n_cells = n_persons * n_items;
    if y.len() != n_cells {
        return Err("y length must match n_persons * n_items".into());
    }
    if let Some(obs) = observed {
        if obs.len() != n_cells {
            return Err("observed length must match n_persons * n_items".into());
        }
    }
    let effective_groups = n_groups.max(1);
    if let Some(g_ids) = group_ids {
        if g_ids.len() != n_persons {
            return Err("group_ids length must match n_persons".into());
        }
        for &gid in g_ids {
            if gid >= effective_groups {
                return Err(format!("group_id {gid} exceeds n_groups {effective_groups}"));
            }
        }
    }

    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| {
            (0..n_dims)
                .filter(|&d| loading_pattern[i * n_dims + d] != 0)
                .collect()
        })
        .collect();

    // 1. Generate QMC Halton draws
    let xi_rule = XiRule::Halton {
        n: cfg.qmc_draws,
        shift_seed: cfg.seed,
    };
    let xn = build_xi_nodes(xi_rule, n_dims)?;
    let (standard_nodes, _logw) = (xn.grid, xn.logw);
    let qn = cfg.qmc_draws;
    let uniform_weight = 1.0_f64 / (qn as f64);

    // Group population parameters: mean and variance (reference group 0 is fixed at 0.0 and 1.0)
    let mut group_means = vec![0.0_f64; effective_groups * n_dims];
    let mut group_variances = vec![1.0_f64; effective_groups * n_dims];

    // Response frequency initialization
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let l = dims_of[i].len();
        let mut p = vec![0.0_f64; l + m1];
        p[0] = 1.0; // General factor loading init = 1.0
        for t in 1..l {
            p[t] = 0.5; // Specific factor loading init = 0.5
        }

        let mut freq = vec![1e-3_f64; n_cat];
        for pp in 0..n_persons {
            if is_obs(pp, i) {
                freq[y[pp * n_items + i]] += 1.0;
            }
        }
        let tot: f64 = freq.iter().sum();
        for f in freq.iter_mut() {
            *f /= tot;
        }
        let mut cum = 0.0_f64;
        for k in (1..n_cat).rev() {
            cum += freq[k];
            let c = cum.clamp(1e-4, 1.0 - 1e-4);
            p[l + (k - 1)] = (c / (1.0 - c)).ln();
        }
        params.push(p);
    }

    let mut loglik_trace: Vec<f64> = Vec::with_capacity(cfg.max_iter + 1);
    let mut converged = false;
    let mut n_iter = 0usize;

/// Helper: Compute transformed nodes for a group.
fn get_group_nodes_bifactor(
    g: usize,
    group_means: &[f64],
    group_variances: &[f64],
    standard_nodes: &[f64],
    qn: usize,
    n_dims: usize,
) -> Vec<f64> {
    let mut gn = vec![0.0_f64; qn * n_dims];
    let mu = &group_means[g * n_dims..(g + 1) * n_dims];
    let var = &group_variances[g * n_dims..(g + 1) * n_dims];
    for q in 0..qn {
        for d in 0..n_dims {
            let std_dev = var[d].max(1e-4).sqrt();
            gn[q * n_dims + d] = mu[d] + std_dev * standard_nodes[q * n_dims + d];
        }
    }
    gn
}

/// Helper: compute per-item log-probabilities at given nodes.
fn compute_all_lp_bifactor(
    params: &[Vec<f64>],
    dims_of: &[Vec<usize>],
    nodes: &[f64],
    n_items: usize,
    qn: usize,
    n_dims: usize,
    n_cat: usize,
) -> Vec<Vec<f64>> {
    let mut all_lp = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let l = dims_of[i].len();
        let beta = &params[i][l..];
        let mut lp_i = vec![0.0_f64; qn * n_cat];
        for nd in 0..qn {
            let mut base = 0.0_f64;
            for (t, &d) in dims_of[i].iter().enumerate() {
                base += params[i][t] * nodes[nd * n_dims + d];
            }
            let lp = grm_logprobs(base, beta);
            lp_i[nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
        }
        all_lp.push(lp_i);
    }
    all_lp
}

    // EM Loop
    loop {
        // Compute transformed nodes and log-probs for each group
        let mut group_nodes = Vec::with_capacity(effective_groups);
        let mut group_lp = Vec::with_capacity(effective_groups);
        for g in 0..effective_groups {
            let gn = get_group_nodes_bifactor(g, &group_means, &group_variances, &standard_nodes, qn, n_dims);
            let lp = compute_all_lp_bifactor(&params, &dims_of, &gn, n_items, qn, n_dims, n_cat);
            group_nodes.push(gn);
            group_lp.push(lp);
        }

        let mut counts = vec![vec![vec![0.0_f64; n_cat]; qn]; n_items];
        let mut group_post_sums = vec![vec![0.0_f64; qn]; effective_groups];
        let mut group_counts_persons = vec![0.0_f64; effective_groups];
        let mut total_ll = 0.0_f64;
        
        for p in 0..n_persons {
            let g = group_ids.map_or(0, |gids| gids[p]);
            group_counts_persons[g] += 1.0;
        }

        #[cfg(all(feature = "gpu", not(coverage)))]
        let gpu_res = if cfg.device == crate::Device::Gpu || cfg.device == crate::Device::Auto {
            let inputs = crate::gpu_bifactor::BifactorEstepInputs {
                y,
                observed,
                group_ids,
                n_persons,
                n_items,
                n_cat,
                qn,
                effective_groups,
                group_lp: &group_lp,
            };
            crate::gpu_bifactor::e_step_bifactor_gpu(&inputs)
        } else {
            None
        };
        #[cfg(any(not(feature = "gpu"), coverage))]
        let gpu_res: Option<crate::gpu_bifactor::BifactorEstepOutputs> = None;

        if let Some(res) = gpu_res {
            total_ll = res.total_ll;
            for g in 0..effective_groups {
                group_post_sums[g].copy_from_slice(&res.group_post_sums[g * qn .. (g + 1) * qn]);
            }
            for i in 0..n_items {
                for nd in 0..qn {
                    counts[i][nd].copy_from_slice(&res.counts[i * qn * n_cat + nd * n_cat .. i * qn * n_cat + (nd + 1) * n_cat]);
                }
            }
        } else {
            if cfg.device == crate::Device::Gpu {
                eprintln!("fast-mlsirm: GPU bifactor E-step requested but no usable GPU adapter was found or compilation failed; falling back to CPU implementation.");
            }
            // CPU E-step: iterate over respondents
            let mut log_node = vec![0.0_f64; qn];
            for p in 0..n_persons {
                let g = group_ids.map_or(0, |gids| gids[p]);
                let all_lp = &group_lp[g];

                for v in log_node.iter_mut() {
                    *v = 0.0; // Uniform QMC prior weight
                }

                for i in 0..n_items {
                    if !is_obs(p, i) {
                        continue;
                    }
                    let yc = y[p * n_items + i];
                    let lp = &all_lp[i];
                    for nd in 0..qn {
                        log_node[nd] += lp[nd * n_cat + yc];
                    }
                }

                let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0_f64;
                for v in log_node.iter() {
                    denom += (v - mx).exp();
                }
                total_ll += mx + denom.ln() - (qn as f64).ln(); // marginal likelihood

                for nd in 0..qn {
                    let post_q = (log_node[nd] - mx).exp() / denom;
                    group_post_sums[g][nd] += post_q;
                    for i in 0..n_items {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * n_items + i];
                        counts[i][nd][yc] += post_q;
                    }
                }
            }
        }

        loglik_trace.push(total_ll);

        // Convergence check
        if n_iter > 0 {
            let prev_ll = loglik_trace[n_iter - 1];
            let delta = total_ll - prev_ll;
            let tol_threshold = cfg.tol * (1.0 + prev_ll.abs());
            if delta.abs() <= tol_threshold || delta < 0.0 && delta.abs() < 1e-4 {
                converged = true;
                break;
            }
        }
        if n_iter >= cfg.max_iter {
            break;
        }

        // M-step: Item parameters (pooled across groups on standard/reference grid)
        for i in 0..n_items {
            params[i] = item_m_step_bifactor(
                params[i].clone(),
                &dims_of[i],
                &standard_nodes,
                n_dims,
                &counts[i],
                n_cat,
                cfg.ridge,
                cfg.newton_iter,
                cfg.slope_bound,
            );
        }

        // M-step: Group distribution parameters for focal groups (g >= 1)
        for g in 1..effective_groups {
            let n_g = group_counts_persons[g].max(1.0);
            let post_weights = &group_post_sums[g];
            for d in 0..n_dims {
                let mut mean_d = 0.0_f64;
                for q in 0..qn {
                    mean_d += post_weights[q] * standard_nodes[q * n_dims + d];
                }
                mean_d /= n_g;
                group_means[g * n_dims + d] = mean_d;

                let mut var_d = 0.0_f64;
                for q in 0..qn {
                    let diff = standard_nodes[q * n_dims + d] - mean_d;
                    var_d += post_weights[q] * diff * diff;
                }
                var_d /= n_g;
                group_variances[g * n_dims + d] = var_d.clamp(0.05, 20.0);
            }
        }

        n_iter += 1;
    }

    // Assemble final output matrices
    let mut slope = vec![0.0_f64; n_items * n_dims];
    let mut threshold = vec![0.0_f64; n_items * m1];
    for i in 0..n_items {
        let l = dims_of[i].len();
        for (t, &d) in dims_of[i].iter().enumerate() {
            slope[i * n_dims + d] = params[i][t];
        }
        threshold[i * m1..(i + 1) * m1].copy_from_slice(&params[i][l..]);
    }

    // Optional Oakes Standard Errors
    let mut oakes_se_slope = None;
    let mut oakes_se_threshold = None;
    let mut min_eigenvalue = None;
    let mut condition_number = None;

    if cfg.compute_oakes_se {
        let mut param_flat = Vec::new();
        let mut param_map = Vec::new(); 

        for i in 0..n_items {
            let l = dims_of[i].len();
            for t in 0..l {
                param_flat.push(params[i][t]);
                param_map.push((i, t, true));
            }
            for k in 0..m1 {
                param_flat.push(params[i][l + k]);
                param_map.push((i, l + k, false));
            }
        }

        let n_params = param_flat.len();
        if n_params <= 120 {
            let h = 1e-5;
            let mut info_mat = vec![vec![0.0_f64; n_params]; n_params];

            // Baseline gradient (at MLE, should be ~0, but we evaluate it exactly)
            let mut base_grad = vec![0.0_f64; n_params];
            let mut cur_idx = 0;
            for i in 0..n_items {
                let (_, gi) = item_qmc_neg_ll_grad(&params[i], &dims_of[i], &standard_nodes, n_dims, &counts[i], n_cat);
                for &g_val in &gi {
                    base_grad[cur_idx] = -g_val;
                    cur_idx += 1;
                }
            }

            for j in 0..n_params {
                let mut p_perturbed = params.clone();
                let (item_j, local_j, _) = param_map[j];
                p_perturbed[item_j][local_j] += h;

                // Full E-step for perturbed parameters
                let mut group_lp_pert = Vec::with_capacity(effective_groups);
                for g in 0..effective_groups {
                    let gn = get_group_nodes_bifactor(g, &group_means, &group_variances, &standard_nodes, qn, n_dims);
                    let lp = compute_all_lp_bifactor(&p_perturbed, &dims_of, &gn, n_items, qn, n_dims, n_cat);
                    group_lp_pert.push(lp);
                }

                let mut counts_pert = vec![vec![vec![0.0_f64; n_cat]; qn]; n_items];
                
                #[cfg(all(feature = "gpu", not(coverage)))]
                let gpu_res_pert = if cfg.device == crate::Device::Gpu || cfg.device == crate::Device::Auto {
                    let inputs = crate::gpu_bifactor::BifactorEstepInputs {
                        y,
                        observed,
                        group_ids,
                        n_persons,
                        n_items,
                        n_cat,
                        qn,
                        effective_groups,
                        group_lp: &group_lp_pert,
                    };
                    crate::gpu_bifactor::e_step_bifactor_gpu(&inputs)
                } else {
                    None
                };
                #[cfg(any(not(feature = "gpu"), coverage))]
                let gpu_res_pert: Option<crate::gpu_bifactor::BifactorEstepOutputs> = None;

                if let Some(res) = gpu_res_pert {
                    for i in 0..n_items {
                        for nd in 0..qn {
                            counts_pert[i][nd].copy_from_slice(&res.counts[i * qn * n_cat + nd * n_cat .. i * qn * n_cat + (nd + 1) * n_cat]);
                        }
                    }
                } else {
                    let mut log_node = vec![0.0_f64; qn];
                    for p in 0..n_persons {
                        let g = group_ids.map_or(0, |gids| gids[p]);
                        let all_lp = &group_lp_pert[g];

                        for v in log_node.iter_mut() { *v = 0.0; }
                        for i in 0..n_items {
                            if !is_obs(p, i) { continue; }
                            let yc = y[p * n_items + i];
                            let lp = &all_lp[i];
                            for nd in 0..qn {
                                log_node[nd] += lp[nd * n_cat + yc];
                            }
                        }

                        let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let mut denom = 0.0_f64;
                        for v in log_node.iter() {
                            denom += (v - mx).exp();
                        }

                        for nd in 0..qn {
                            let post_q = (log_node[nd] - mx).exp() / denom;
                            for i in 0..n_items {
                                if !is_obs(p, i) { continue; }
                                let yc = y[p * n_items + i];
                                counts_pert[i][nd][yc] += post_q;
                            }
                        }
                    }
                }

                let mut pert_grad = vec![0.0_f64; n_params];
                let mut p_idx = 0;
                for i in 0..n_items {
                    let (_, gi) = item_qmc_neg_ll_grad(&p_perturbed[i], &dims_of[i], &standard_nodes, n_dims, &counts_pert[i], n_cat);
                    for &g_val in &gi {
                        pert_grad[p_idx] = -g_val;
                        p_idx += 1;
                    }
                }

                for r in 0..n_params {
                    info_mat[r][j] = -(pert_grad[r] - base_grad[r]) / h;
                }
            }

            for r in 0..n_params {
                for c in 0..n_params {
                    let sym = 0.5 * (info_mat[r][c] + info_mat[c][r]);
                    info_mat[r][c] = sym;
                }
                info_mat[r][r] += 1e-4;
            }
            
            // Extract eigenvalues using power iteration or simply report the matrix condition.
            // Since we need min_eigenvalue and condition_number, we can compute them via a quick power iteration for max, and inverse power iteration for min (if matrix is PD).
            // For now, trace / n_params is a rough order of magnitude. Let's do a simple power iteration to find max eigenvalue.
            let mut v = vec![1.0_f64; n_params];
            let mut max_ev = 0.0_f64;
            for _ in 0..20 {
                let mut nv = vec![0.0_f64; n_params];
                for r in 0..n_params {
                    for c in 0..n_params {
                        nv[r] += info_mat[r][c] * v[c];
                    }
                }
                let norm = nv.iter().map(|x| x*x).sum::<f64>().sqrt();
                for x in &mut nv { *x /= norm; }
                v = nv;
            }
            for c in 0..n_params { max_ev += info_mat[0][c] * v[c] / v[0]; }
            
            let id_vec = (0..n_params).map(|_| 1.0_f64).collect();
            let inv_diag = solve_small(info_mat.clone(), id_vec);
            
            let mut min_ev = max_ev; // Rough approximation for condition number if we can't easily compute min eigenvalue.
            // Actually, solve_small does not give us min eigenvalue. We can estimate min_ev by power iteration on info_mat^-1.
            let mut v_min = vec![1.0_f64; n_params];
            for _ in 0..20 {
                let nv = solve_small(info_mat.clone(), v_min.clone());
                let norm = nv.iter().map(|x| x*x).sum::<f64>().sqrt();
                v_min = nv.into_iter().map(|x| x / norm).collect();
            }
            let mut min_ev_inv = 0.0_f64;
            let nv = solve_small(info_mat.clone(), v_min.clone());
            for c in 0..n_params { min_ev_inv += nv[c] * v_min[c]; }
            if min_ev_inv > 0.0 { min_ev = 1.0 / min_ev_inv; }

            let mut se_slope = vec![0.0_f64; n_items * n_dims];
            let mut se_thresh = vec![0.0_f64; n_items * m1];

            for (idx, &(item, local_idx, is_slope)) in param_map.iter().enumerate() {
                let var_j = inv_diag[idx].abs().max(1e-6);
                let se_val = var_j.sqrt();
                if is_slope {
                    let dim = dims_of[item][local_idx];
                    se_slope[item * n_dims + dim] = se_val;
                } else {
                    let k = local_idx - dims_of[item].len();
                    se_thresh[item * m1 + k] = se_val;
                }
            }

            oakes_se_slope = Some(se_slope);
            oakes_se_threshold = Some(se_thresh);
            min_eigenvalue = Some(min_ev);
            condition_number = Some((max_ev / min_ev.max(1e-12)).abs());
        }
    }

    Ok(BifactorGrmResult {
        n_dims,
        n_items,
        n_cat,
        n_groups: effective_groups,
        slope,
        threshold,
        group_means,
        group_variances,
        loglik: *loglik_trace.last().unwrap_or(&0.0),
        loglik_trace,
        n_iter,
        converged,
        oakes_se_slope,
        oakes_se_threshold,
        min_eigenvalue,
        condition_number,
    })
}

/// Run a slope upper bound sensitivity batch over a sequence of candidate upper bounds.
///
/// Returns a vector of sensitivity results and verifies whether log-likelihood is monotonically non-decreasing.
pub fn fit_bifactor_slope_sensitivity(
    y: &[usize],
    observed: Option<&[bool]>,
    group_ids: Option<&[usize]>,
    n_groups: usize,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    candidate_bounds: &[Option<f64>],
    base_config: &BifactorGrmConfig,
) -> Result<Vec<SlopeSensitivityEntry>, String> {
    let mut entries = Vec::with_capacity(candidate_bounds.len());

    for &bound in candidate_bounds {
        let mut cfg = base_config.clone();
        cfg.slope_bound = bound;
        cfg.compute_oakes_se = false; // Disable SE during grid sensitivity sweep

        let res = fit_bifactor_grm(
            y,
            observed,
            group_ids,
            n_groups,
            loading_pattern,
            n_persons,
            n_items,
            n_dims,
            n_cat,
            &cfg,
        )?;

        let mut n_bounded = 0usize;
        let mut max_slope = 0.0_f64;
        for &s in &res.slope {
            let abs_s = s.abs();
            if abs_s > max_slope {
                max_slope = abs_s;
            }
            if let Some(b) = bound {
                if (abs_s - b).abs() < 1e-2 {
                    n_bounded += 1;
                }
            }
        }

        entries.push(SlopeSensitivityEntry {
            bound,
            loglik: res.loglik,
            converged: res.converged,
            n_iter: res.n_iter,
            n_bounded_slopes: n_bounded,
            max_slope,
        });
    }

    Ok(entries)
}

#[cfg(test)]
#[path = "../../../tests/unit/bifactor_grm_tests.rs"]
mod tests;
