//! Two-stage Lord-Wingersky recursion for bifactor models (Gibbons & Hedeker 1992; Cai 2011).
//!
//! In a bifactor model, items are conditionally independent given the general
//! factor theta_0 and the specific factor theta_s of each item's domain.
//!
//! Two-stage recursion:
//! 1. Stage 1 (Within-Domain): For each specific domain s and each general trait node theta_0,
//!    run Lord-Wingersky recursion over items in domain s conditional on (theta_0, theta_s),
//!    then integrate out theta_s using the specific factor quadrature rule. This yields the
//!    conditional sum-score distribution for domain s: P(X_s = k | theta_0).
//! 2. Stage 2 (Across-Domain): Given theta_0, domain sum-scores X_1, ..., X_S are conditionally
//!    independent. Convolve P(X_1 | theta_0), ..., P(X_S | theta_0) across domains to obtain
//!    the complete total score distribution P(X = r | theta_0).
//!
//! Reference:
//! - Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis.
//!   Psychometrika, 57(3), 423-436.
//! - Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis.
//!   Psychological Methods, 16(3), 221-248.
//! - Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile equating.
//!   Applied Psychological Measurement, 8(4), 453-461.

/// Specification of item parameters in a bifactor Graded Response Model (GRM).
#[derive(Clone, Debug)]
pub struct BifactorItemParams {
    /// General factor slope for each item (length `n_items`).
    pub a_general: Vec<f64>,
    /// Specific factor slope for each item (length `n_items`).
    pub a_specific: Vec<f64>,
    /// Boundary intercepts beta_ik for each item (length `n_items * (n_cat - 1)`),
    /// ordered decreasing within each item.
    pub thresholds: Vec<f64>,
    /// Domain identifier for each item in `0..n_domains` (length `n_items`).
    pub item_domains: Vec<usize>,
    /// Number of response categories per item (e.g. 2 for dichotomous, 4 for 4-category).
    pub n_cat: usize,
    /// Number of specific domains.
    pub n_domains: usize,
}

impl BifactorItemParams {
    pub fn n_items(&self) -> usize {
        self.a_general.len()
    }

    pub fn total_max_score(&self) -> usize {
        self.n_items() * (self.n_cat - 1)
    }

    /// Compute category response probabilities for item `item_idx` at `(theta_0, theta_s)`.
    /// Returns a slice/array of length `n_cat` summing to 1.0.
    #[inline]
    pub fn category_probabilities(&self, item_idx: usize, theta_0: f64, theta_s: f64) -> Vec<f64> {
        let m1 = self.n_cat - 1;
        let a_g = self.a_general[item_idx];
        let a_s = self.a_specific[item_idx];
        let eta_base = a_g * theta_0 + a_s * theta_s;
        let beta = &self.thresholds[item_idx * m1..(item_idx + 1) * m1];

        // P(Y >= k) = sigmoid(eta_base + beta_{k-1}) for k = 1..m1
        let mut p_cum = vec![0.0_f64; m1 + 1];
        p_cum[0] = 1.0; // P(Y >= 0) = 1.0
        for k in 1..=m1 {
            let z = eta_base + beta[k - 1];
            p_cum[k] = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            };
        }

        // P(Y = k) = P(Y >= k) - P(Y >= k + 1)
        let mut probs = vec![0.0_f64; self.n_cat];
        for k in 0..m1 {
            let diff = p_cum[k] - p_cum[k + 1];
            probs[k] = diff.max(0.0);
        }
        probs[m1] = p_cum[m1].max(0.0);

        // Normalize to ensure exact sum to 1.0 despite tiny floating point rounding
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
        probs
    }
}

/// Two-stage Lord-Wingersky recursion for a bifactor model.
///
/// Computes `P(X = r | theta_0)` for `r in 0..=total_max_score` for each point in `theta_general`.
///
/// Returns a row-major matrix of size `n_general_nodes x (total_max_score + 1)`.
pub fn bifactor_lord_wingersky(
    params: &BifactorItemParams,
    theta_general: &[f64],
    theta_specific: &[f64],
    weights_specific: &[f64],
) -> Result<Vec<f64>, String> {
    let n_items = params.n_items();
    let n_cat = params.n_cat;
    let m1 = n_cat - 1;
    let n_g = theta_general.len();
    let n_s = theta_specific.len();

    if weights_specific.len() != n_s {
        return Err("weights_specific length must match theta_specific length".to_string());
    }
    if n_cat < 2 {
        return Err("n_cat must be at least 2".to_string());
    }
    if params.a_specific.len() != n_items || params.item_domains.len() != n_items {
        return Err("Parameter vector lengths do not match n_items".to_string());
    }
    if params.thresholds.len() != n_items * m1 {
        return Err("thresholds length does not match n_items * (n_cat - 1)".to_string());
    }

    let total_max_score = params.total_max_score();
    let mut out = vec![0.0_f64; n_g * (total_max_score + 1)];

    // Group items by domain
    let mut domain_items: Vec<Vec<usize>> = vec![Vec::new(); params.n_domains];
    for (i, &d) in params.item_domains.iter().enumerate() {
        if d >= params.n_domains {
            return Err(format!("item_domains contains invalid domain {d} >= n_domains {}", params.n_domains));
        }
        domain_items[d].push(i);
    }

    // For each general node theta_0
    for (g_idx, &th0) in theta_general.iter().enumerate() {
        // Stage 1: Within-domain recursion for each domain
        let mut domain_score_dists: Vec<Vec<f64>> = Vec::with_capacity(params.n_domains);

        for items in &domain_items {
            let domain_max = items.len() * m1;
            if items.is_empty() {
                domain_score_dists.push(vec![1.0]);
                continue;
            }

            let mut p_domain = vec![0.0_f64; domain_max + 1];

            // Integrate over specific factor nodes
            for (s_idx, &ths) in theta_specific.iter().enumerate() {
                let w = weights_specific[s_idx];
                if w == 0.0 {
                    continue;
                }

                // Run Lord-Wingersky polytomous convolution for items in this domain
                let mut f = vec![0.0_f64; domain_max + 1];
                f[0] = 1.0;
                let mut current_max = 0usize;

                for &item_idx in items {
                    let cat_probs = params.category_probabilities(item_idx, th0, ths);
                    let next_max = current_max + m1;
                    let mut f_next = vec![0.0_f64; domain_max + 1];

                    for r in 0..=current_max {
                        let prev_p = f[r];
                        if prev_p == 0.0 {
                            continue;
                        }
                        for k in 0..n_cat {
                            f_next[r + k] += prev_p * cat_probs[k];
                        }
                    }

                    f = f_next;
                    current_max = next_max;
                }

                // Accumulate weighted distribution
                for r in 0..=domain_max {
                    p_domain[r] += w * f[r];
                }
            }

            domain_score_dists.push(p_domain);
        }

        // Stage 2: Across-domain convolution
        let mut total_dist = vec![1.0_f64];
        let mut current_max_total = 0usize;

        for p_domain in domain_score_dists {
            let domain_max = p_domain.len() - 1;
            let next_max_total = current_max_total + domain_max;
            let mut next_dist = vec![0.0_f64; next_max_total + 1];

            for r1 in 0..=current_max_total {
                let p1 = total_dist[r1];
                if p1 == 0.0 {
                    continue;
                }
                for r2 in 0..=domain_max {
                    next_dist[r1 + r2] += p1 * p_domain[r2];
                }
            }

            total_dist = next_dist;
            current_max_total = next_max_total;
        }

        // Copy into output matrix row
        let row_offset = g_idx * (total_max_score + 1);
        for r in 0..=total_max_score {
            if r <= current_max_total {
                out[row_offset + r] = total_dist[r];
            } else {
                out[row_offset + r] = 0.0;
            }
        }
    }

    Ok(out)
}

/// Direct enumeration of all response vectors to compute the exact ground truth
/// P(X = r | theta_0) for verification against the 2-stage Lord-Wingersky recursion.
///
/// Exponential complexity O(n_cat^n_items * n_specific_nodes * n_general_nodes).
/// Only suitable for small item counts (e.g. 4-8 items), exactly as used in verification tests.
pub fn direct_enumeration_bifactor(
    params: &BifactorItemParams,
    theta_general: &[f64],
    theta_specific: &[f64],
    weights_specific: &[f64],
) -> Result<Vec<f64>, String> {
    let n_items = params.n_items();
    let n_cat = params.n_cat;
    let n_g = theta_general.len();
    let n_s = theta_specific.len();
    if weights_specific.len() != n_s {
        return Err("weights_specific length must match theta_specific length".to_string());
    }

    if n_items > 12 {
        return Err(format!("Direct enumeration requested for {n_items} items, which would exceed combinatorial limits"));
    }

    let total_max_score = params.total_max_score();
    let mut out = vec![0.0_f64; n_g * (total_max_score + 1)];

    let mut domain_items: Vec<Vec<usize>> = vec![Vec::new(); params.n_domains];
    for (i, &d) in params.item_domains.iter().enumerate() {
        domain_items[d].push(i);
    }

    let mut total_patterns = 1usize;
    for _ in 0..n_items {
        total_patterns *= n_cat;
    }

    for (g_idx, &th0) in theta_general.iter().enumerate() {
        let row_offset = g_idx * (total_max_score + 1);

        for pattern_idx in 0..total_patterns {
            let mut pattern = vec![0usize; n_items];
            let mut rem = pattern_idx;
            let mut score = 0usize;
            for i in 0..n_items {
                let cat = rem % n_cat;
                pattern[i] = cat;
                score += cat;
                rem /= n_cat;
            }

            let mut pattern_prob = 1.0_f64;

            for items in &domain_items {
                if items.is_empty() {
                    continue;
                }
                let mut domain_integral = 0.0_f64;

                for (s_idx, &ths) in theta_specific.iter().enumerate() {
                    let w = weights_specific[s_idx];
                    if w == 0.0 {
                        continue;
                    }
                    let mut joint_item_p = 1.0_f64;
                    for &item_idx in items {
                        let cat_p = params.category_probabilities(item_idx, th0, ths);
                        joint_item_p *= cat_p[pattern[item_idx]];
                    }
                    domain_integral += w * joint_item_p;
                }

                pattern_prob *= domain_integral;
            }

            out[row_offset + score] += pattern_prob;
        }
    }

    Ok(out)
}

#[cfg(test)]
#[path = "../../../tests/unit/bifactor_recursion_tests.rs"]
mod tests;
