//! Two-stage Lord-Wingersky recursion for polytomous two-tier GRM models.
//!
//! Extends the bifactor recursion (Gibbons & Hedeker, 1992; Cai et al., 2011)
//! to multiple primary dimensions with plug-in primary trait levels. Each
//! item's linear predictor is ``sum_p a_ip * theta_p + a_S * theta_S``; at
//! scoring time the caller supplies fixed primary coordinates (typically
//! ``TwoTierGrmResult.theta_p_eap``) and the specific tier is integrated out
//! within each item block via Gauss-Hermite quadrature, then domains are
//! convolved (Lord & Wingersky, 1984).
//!
//! References (APA 7th ed.):
//!
//! Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and
//! equipercentile observed-score "equatings." *Applied Psychological
//! Measurement, 8*(4), 453-461. https://doi.org/10.1177/014662168400800409
//!
//! Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//! analysis. *Psychometrika, 57*(3), 423-436.
//! https://doi.org/10.1007/BF02295430
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350

/// Item parameters for two-tier graded Lord-Wingersky recursion.
#[derive(Clone, Debug)]
pub struct TwoTierItemParams {
    /// Primary slopes, row-major ``n_items * n_primary``.
    pub a_primary: Vec<f64>,
    /// Specific slopes, length ``n_items`` (``0.0`` for specific-free items).
    pub a_specific: Vec<f64>,
    /// Boundary intercepts, row-major ``n_items * (n_cat - 1)``.
    pub thresholds: Vec<f64>,
    /// Specific block per item: ``-1`` = specific-free, else ``0..n_specific-1``.
    pub specific_map: Vec<i32>,
    pub n_primary: usize,
    pub n_specific: usize,
    pub n_cat: usize,
}

impl TwoTierItemParams {
    pub fn n_items(&self) -> usize {
        self.a_specific.len()
    }

    pub fn total_max_score(&self) -> usize {
        self.n_items() * (self.n_cat - 1)
    }

    #[inline]
    fn eta(&self, item_idx: usize, theta_primary: &[f64], theta_s: f64) -> f64 {
        let p = self.n_primary;
        let mut acc = 0.0_f64;
        for d in 0..p {
            acc += self.a_primary[item_idx * p + d] * theta_primary[d];
        }
        acc + self.a_specific[item_idx] * theta_s
    }

    /// Category probabilities for item ``item_idx`` at fixed primary levels.
    pub fn category_probabilities(
        &self,
        item_idx: usize,
        theta_primary: &[f64],
        theta_s: f64,
    ) -> Vec<f64> {
        let m1 = self.n_cat - 1;
        let eta_base = self.eta(item_idx, theta_primary, theta_s);
        let beta = &self.thresholds[item_idx * m1..(item_idx + 1) * m1];

        let mut p_cum = vec![0.0_f64; m1 + 1];
        p_cum[0] = 1.0;
        for k in 1..=m1 {
            let z = eta_base + beta[k - 1];
            p_cum[k] = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            };
        }

        let mut probs = vec![0.0_f64; self.n_cat];
        for k in 0..m1 {
            probs[k] = (p_cum[k] - p_cum[k + 1]).max(0.0);
        }
        probs[m1] = p_cum[m1].max(0.0);

        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
        probs
    }

    fn block_items(&self) -> Result<Vec<Vec<usize>>, String> {
        let n_items = self.n_items();
        if self.a_primary.len() != n_items * self.n_primary {
            return Err("a_primary length must be n_items * n_primary".into());
        }
        let m1 = self.n_cat - 1;
        if self.thresholds.len() != n_items * m1 {
            return Err("thresholds length must be n_items * (n_cat - 1)".into());
        }
        if self.specific_map.len() != n_items {
            return Err("specific_map length must equal n_items".into());
        }

        // One block per specific factor plus a trailing block for specific-free items.
        let n_blocks = self.n_specific + 1;
        let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
        for (i, &s) in self.specific_map.iter().enumerate() {
            let block = if s < 0 {
                self.n_specific
            } else if s as usize >= self.n_specific {
                return Err(format!(
                    "specific_map[{i}] = {s} out of range 0..{}",
                    self.n_specific - 1
                ));
            } else {
                s as usize
            };
            blocks[block].push(i);
        }
        Ok(blocks)
    }
}

/// Total-score distribution ``P(X = r | theta_primary)`` for each primary row.
///
/// ``theta_primary`` is row-major ``n_persons * n_primary``. Returns row-major
/// ``n_persons * (total_max_score + 1)``.
pub fn two_tier_lord_wingersky(
    params: &TwoTierItemParams,
    theta_primary: &[f64],
    theta_specific: &[f64],
    weights_specific: &[f64],
) -> Result<Vec<f64>, String> {
    let n_items = params.n_items();
    let n_cat = params.n_cat;
    let m1 = n_cat - 1;
    let p = params.n_primary;
    if p < 1 {
        return Err("n_primary must be >= 1".into());
    }
    if n_cat < 2 {
        return Err("n_cat must be at least 2".into());
    }
    if n_items == 0 {
        return Err("n_items must be >= 1".into());
    }
    if theta_primary.len() % p != 0 {
        return Err("theta_primary length must be a multiple of n_primary".into());
    }
    let n_persons = theta_primary.len() / p;
    let n_s = theta_specific.len();
    if weights_specific.len() != n_s {
        return Err("weights_specific length must match theta_specific".into());
    }

    let blocks = params.block_items()?;
    let total_max_score = params.total_max_score();
    let mut out = vec![0.0_f64; n_persons * (total_max_score + 1)];

    for person in 0..n_persons {
        let th_p = &theta_primary[person * p..(person + 1) * p];
        let mut block_dists: Vec<Vec<f64>> = Vec::with_capacity(blocks.len());

        for (block_idx, items) in blocks.iter().enumerate() {
            let domain_max = items.len() * m1;
            if items.is_empty() {
                block_dists.push(vec![1.0]);
                continue;
            }

            let mut p_block = vec![0.0_f64; domain_max + 1];
            let specific_free = block_idx == params.n_specific;

            if specific_free {
                let mut f = vec![0.0_f64; domain_max + 1];
                f[0] = 1.0;
                let mut current_max = 0usize;
                for &item_idx in items {
                    let cat_probs = params.category_probabilities(item_idx, th_p, 0.0);
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
                p_block.copy_from_slice(&f);
            } else {
                for (s_idx, &ths) in theta_specific.iter().enumerate() {
                    let w = weights_specific[s_idx];
                    if w == 0.0 {
                        continue;
                    }
                    let mut f = vec![0.0_f64; domain_max + 1];
                    f[0] = 1.0;
                    let mut current_max = 0usize;
                    for &item_idx in items {
                        let cat_probs = params.category_probabilities(item_idx, th_p, ths);
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
                    for r in 0..=domain_max {
                        p_block[r] += w * f[r];
                    }
                }
            }
            block_dists.push(p_block);
        }

        let mut total_dist = vec![1.0_f64];
        let mut current_max_total = 0usize;
        for p_block in block_dists {
            let block_max = p_block.len() - 1;
            let next_max_total = current_max_total + block_max;
            let mut next_dist = vec![0.0_f64; next_max_total + 1];
            for r1 in 0..=current_max_total {
                let p1 = total_dist[r1];
                if p1 == 0.0 {
                    continue;
                }
                for r2 in 0..=block_max {
                    next_dist[r1 + r2] += p1 * p_block[r2];
                }
            }
            total_dist = next_dist;
            current_max_total = next_max_total;
        }

        let row_offset = person * (total_max_score + 1);
        for r in 0..=total_max_score {
            out[row_offset + r] = if r <= current_max_total {
                total_dist[r]
            } else {
                0.0
            };
        }
    }

    Ok(out)
}

/// Expected raw totals with caller-owned specific-factor quadrature count.
pub fn two_tier_expected_raw_at_q(
    params: &TwoTierItemParams,
    theta_primary: &[f64],
    q_specific: usize,
) -> Result<Vec<f64>, String> {
    let (nodes, weights) = crate::quadrature::gh_rule(q_specific).ok_or_else(|| {
        format!("unsupported specific-factor quadrature count {q_specific}")
    })?;
    two_tier_expected_raw(params, theta_primary, nodes, weights)
}

/// Expected raw total score at plug-in primary coordinates (Lord-Wingersky mean).
pub fn two_tier_expected_raw(
    params: &TwoTierItemParams,
    theta_primary: &[f64],
    theta_specific: &[f64],
    weights_specific: &[f64],
) -> Result<Vec<f64>, String> {
    let p = params.n_primary;
    if theta_primary.len() % p != 0 {
        return Err("theta_primary length must be a multiple of n_primary".into());
    }
    let n_persons = theta_primary.len() / p;
    let dist = two_tier_lord_wingersky(params, theta_primary, theta_specific, weights_specific)?;
    let total_max_score = params.total_max_score();
    let mut out = vec![0.0_f64; n_persons];
    for person in 0..n_persons {
        let row_offset = person * (total_max_score + 1);
        let mut mean = 0.0_f64;
        for r in 0..=total_max_score {
            mean += (r as f64) * dist[row_offset + r];
        }
        out[person] = mean;
    }
    Ok(out)
}

/// Direct enumeration oracle for small item sets (verification only).
pub fn direct_enumeration_two_tier(
    params: &TwoTierItemParams,
    theta_primary: &[f64],
    theta_specific: &[f64],
    weights_specific: &[f64],
) -> Result<Vec<f64>, String> {
    let n_items = params.n_items();
    let n_cat = params.n_cat;
    let p = params.n_primary;
    if theta_primary.len() % p != 0 {
        return Err("theta_primary length must be a multiple of n_primary".into());
    }
    let n_persons = theta_primary.len() / p;
    let n_s = theta_specific.len();
    if weights_specific.len() != n_s {
        return Err("weights_specific length must match theta_specific".into());
    }
    if n_items > 10 {
        return Err(format!(
            "direct enumeration requested for {n_items} items; limit is 10"
        ));
    }

    let blocks = params.block_items()?;
    let total_max_score = params.total_max_score();
    let mut out = vec![0.0_f64; n_persons * (total_max_score + 1)];

    let mut total_patterns = 1usize;
    for _ in 0..n_items {
        total_patterns *= n_cat;
    }

    for person in 0..n_persons {
        let th_p = &theta_primary[person * p..(person + 1) * p];
        let row_offset = person * (total_max_score + 1);

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
            for (block_idx, items) in blocks.iter().enumerate() {
                if items.is_empty() {
                    continue;
                }
                let specific_free = block_idx == params.n_specific;
                let mut block_integral = 0.0_f64;
                if specific_free {
                    let mut joint = 1.0_f64;
                    for &item_idx in items {
                        let cat_p = params.category_probabilities(item_idx, th_p, 0.0);
                        joint *= cat_p[pattern[item_idx]];
                    }
                    block_integral = joint;
                } else {
                    for (s_idx, &ths) in theta_specific.iter().enumerate() {
                        let w = weights_specific[s_idx];
                        if w == 0.0 {
                            continue;
                        }
                        let mut joint = 1.0_f64;
                        for &item_idx in items {
                            let cat_p = params.category_probabilities(item_idx, th_p, ths);
                            joint *= cat_p[pattern[item_idx]];
                        }
                        block_integral += w * joint;
                    }
                }
                pattern_prob *= block_integral;
            }
            out[row_offset + score] += pattern_prob;
        }
    }

    Ok(out)
}

#[cfg(test)]
#[path = "../../../tests/unit/two_tier_recursion_tests.rs"]
mod tests;
