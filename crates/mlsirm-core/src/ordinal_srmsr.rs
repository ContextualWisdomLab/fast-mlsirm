//! Declared ordinal-score SRMSR (Maydeu-Olivares & Joe, 2014, p. 317, Eqs. 32–35).
//!
//! Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in categorical
//! data analysis. *Multivariate Behavioral Research, 49*(4), 305–328.
//! https://doi.org/10.1080/00273171.2014.911075
//!
//! Maydeu-Olivares and Joe (2014, p. 317, Equation 32) define the residual
//! product-moment correlation as the sample Pearson correlation minus the
//! expected correlation. The expected correlation is the model covariance
//! divided by the model standard deviations, not divided by the sample standard
//! deviations (pp. 317, 320). Equations 33–35 give the second moment and the
//! square root of the mean squared residual correlation. Category scores are
//! `0, 1, ..., K - 1` (Appendix, Equations 25–26, 33, 63–64).
//!
//! Gibbons and Hedeker (1992, pp. 424–428, Equations 2–11) supply the bifactor
//! conditional independence reduction used to integrate shared-`G` nodes.
//! Cai (2010, p. 589, Equations 11–12) supplies the GRM category differences.
//!
//! # References
//! Cai, L. (2010). A two-tier full-information item factor analysis model with
//! applications. *Psychometrika, 75*(4), 581–612.
//! https://doi.org/10.1007/s11336-010-9178-0
//!
//! Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//! analysis. *Psychometrika, 57*(3), 423–436. https://doi.org/10.1007/BF02295430
//!
//! Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in categorical
//! data analysis. *Multivariate Behavioral Research, 49*(4), 305–328.
//! https://doi.org/10.1080/00273171.2014.911075
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairResidual {
    pub group: usize,
    pub item_i: usize,
    pub item_j: usize,
    pub n_pair: usize,
    pub observed_corr: f64,
    pub expected_corr: f64,
    pub residual: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupOrdinalSrmsr {
    pub group: usize,
    pub n_group: usize,
    pub n_pairs: usize,
    pub srmsr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OrdinalSrmsr {
    pub definition: &'static str,
    pub category_scores: &'static str,
    pub group_aggregation: &'static str,
    pub q_general: usize,
    pub q_specific: usize,
    pub n_groups: usize,
    pub n_items: usize,
    pub n_cat: usize,
    pub n_specific: usize,
    pub n_pairs_per_group: usize,
    pub n_total_pairs: usize,
    pub unweighted_srmsr: f64,
    pub group_size_weighted_srmsr: f64,
    pub groups: Vec<GroupOrdinalSrmsr>,
    pub residuals: Vec<PairResidual>,
}

struct Moments {
    mean: Vec<f64>,
    second: Vec<f64>,
    cross: Vec<f64>,
}

fn pair_index(i: usize, j: usize, n_items: usize) -> usize {
    i * (2 * n_items - i - 1) / 2 + (j - i - 1)
}

fn observed_moments(
    y: &[usize],
    observed: Option<&[bool]>,
    groups: &[usize],
    group: usize,
    n_items: usize,
    n_cat: usize,
) -> Result<(Moments, Vec<usize>), String> {
    let mut count = vec![0usize; n_items];
    let mut pair_count = vec![0usize; n_items * (n_items - 1) / 2];
    let mut mean = vec![0.0; n_items];
    let mut second = vec![0.0; n_items];
    let mut cross = vec![0.0; pair_count.len()];
    for (person, &g) in groups.iter().enumerate() {
        if g != group {
            continue;
        }
        let row = person * n_items;
        for i in 0..n_items {
            if observed.is_some_and(|mask| !mask[row + i]) {
                continue;
            }
            let value = y[row + i];
            if value >= n_cat {
                return Err("observed response category out of range".into());
            }
            let score = value as f64;
            count[i] += 1;
            mean[i] += score;
            second[i] += score * score;
        }
        for i in 0..n_items {
            if observed.is_some_and(|mask| !mask[row + i]) {
                continue;
            }
            for j in i + 1..n_items {
                if observed.is_some_and(|mask| !mask[row + j]) {
                    continue;
                }
                let idx = pair_index(i, j, n_items);
                pair_count[idx] += 1;
                cross[idx] += y[row + i] as f64 * y[row + j] as f64;
            }
        }
    }
    if count.iter().any(|&n| n == 0) || pair_count.iter().any(|&n| n == 0) {
        return Err("empty group or unobserved item pair".into());
    }
    for i in 0..n_items {
        mean[i] /= count[i] as f64;
        second[i] /= count[i] as f64;
    }
    for (value, &n) in cross.iter_mut().zip(&pair_count) {
        *value /= n as f64;
    }
    Ok((Moments { mean, second, cross }, pair_count))
}

fn model_moments(
    a_general: &[f64],
    a_specific: &[f64],
    threshold: &[f64],
    specific_map: &[i32],
    n_specific: usize,
    mean_g: f64,
    sd_g: f64,
    mean_s: &[f64],
    sd_s: &[f64],
    n_items: usize,
    n_cat: usize,
    q_general: usize,
    q_specific: usize,
) -> Result<Moments, String> {
    if specific_map.iter().any(|&s| s < -1 || (s >= 0 && s as usize >= n_specific)) {
        return Err("invalid specific_map entry".into());
    }
    for (item, &specific) in specific_map.iter().enumerate() {
        if specific == -1 && a_specific[item] != 0.0 {
            return Err("general-only item must have zero specific slope".into());
        }
    }
    if a_general.iter().chain(a_specific).chain(threshold).chain([&mean_g]).chain(mean_s).any(|v| !v.is_finite()) {
        return Err("parameters must be finite".into());
    }
    if !sd_g.is_finite() || sd_g <= 0.0 || sd_s.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("latent SD must be finite and positive".into());
    }
    let boundaries = n_cat - 1;
    if threshold.chunks_exact(boundaries).any(|row| row.windows(2).any(|w| w[0] <= w[1])) {
        return Err("thresholds must be strictly decreasing".into());
    }
    let (xg, wg) = crate::quadrature::require_gh_rule_unidim(q_general, "ordinal SRMSR general quadrature")?;
    let (xs, ws) = if n_specific > 0 {
        crate::quadrature::require_gh_rule_unidim(q_specific, "ordinal SRMSR specific quadrature")?
    } else {
        (std::slice::from_ref(&0.0), std::slice::from_ref(&1.0))
    };
    let pairs = n_items * (n_items - 1) / 2;
    let mut mean = vec![0.0; n_items];
    let mut second = vec![0.0; n_items];
    let mut cross = vec![0.0; pairs];
    let mut scratch = vec![0.0; n_cat];
    let mut item_m1 = vec![0.0; n_items];
    let mut item_m2 = vec![0.0; n_items];
    let mut node_m1 = vec![0.0; xs.len()];
    for (&node_g, &weight_g) in xg.iter().zip(wg) {
        let theta_g = mean_g + sd_g * node_g;
        for item in 0..n_items {
            let specific = specific_map[item];
            if specific < 0 {
                crate::poly::grm_logprobs_into(
                    a_general[item] * theta_g,
                    &threshold[item * boundaries..(item + 1) * boundaries],
                    &mut scratch,
                );
                let (mut first, mut second_node) = (0.0, 0.0);
                for (category, &log_probability) in scratch.iter().enumerate() {
                    let probability = log_probability.exp();
                    let score = category as f64;
                    first += score * probability;
                    second_node += score * score * probability;
                }
                item_m1[item] = first;
                item_m2[item] = second_node;
            } else {
                let s = specific as usize;
                let (mut first_bar, mut second_bar) = (0.0, 0.0);
                for (&node_s, &weight_s) in xs.iter().zip(ws) {
                    let theta_s = mean_s[s] + sd_s[s] * node_s;
                    crate::poly::grm_logprobs_into(
                        a_general[item] * theta_g + a_specific[item] * theta_s,
                        &threshold[item * boundaries..(item + 1) * boundaries],
                        &mut scratch,
                    );
                    let (mut first, mut second_node) = (0.0, 0.0);
                    for (category, &log_probability) in scratch.iter().enumerate() {
                        let probability = log_probability.exp();
                        let score = category as f64;
                        first += score * probability;
                        second_node += score * score * probability;
                    }
                    first_bar += weight_s * first;
                    second_bar += weight_s * second_node;
                }
                item_m1[item] = first_bar;
                item_m2[item] = second_bar;
            }
            mean[item] += weight_g * item_m1[item];
            second[item] += weight_g * item_m2[item];
        }
        for i in 0..n_items {
            for j in i + 1..n_items {
                let idx = pair_index(i, j, n_items);
                if specific_map[i] >= 0 && specific_map[i] == specific_map[j] {
                    let s = specific_map[i] as usize;
                    for (node_index, &node_s) in xs.iter().enumerate() {
                        let theta_s = mean_s[s] + sd_s[s] * node_s;
                        crate::poly::grm_logprobs_into(
                            a_general[i] * theta_g + a_specific[i] * theta_s,
                            &threshold[i * boundaries..(i + 1) * boundaries],
                            &mut scratch,
                        );
                        node_m1[node_index] = scratch.iter().enumerate().map(|(c, &lp)| c as f64 * lp.exp()).sum();
                    }
                    let mut pair_bar = 0.0;
                    for (node_index, (&node_s, &weight_s)) in xs.iter().zip(ws).enumerate() {
                        let theta_s = mean_s[s] + sd_s[s] * node_s;
                        crate::poly::grm_logprobs_into(
                            a_general[j] * theta_g + a_specific[j] * theta_s,
                            &threshold[j * boundaries..(j + 1) * boundaries],
                            &mut scratch,
                        );
                        let second_item: f64 = scratch.iter().enumerate().map(|(c, &lp)| c as f64 * lp.exp()).sum();
                        pair_bar += weight_s * node_m1[node_index] * second_item;
                    }
                    cross[idx] += weight_g * pair_bar;
                } else {
                    cross[idx] += weight_g * item_m1[i] * item_m1[j];
                }
            }
        }
    }
    Ok(Moments { mean, second, cross })
}

/// Evaluate declared ordinal-score SRMSR for a multigroup bifactor/general-only GRM.
/// `q_specific` is ignored when `n_specific == 0`; it is not a dummy factor.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_multigroup_bifactor_grm(
    y: &[usize],
    observed: Option<&[bool]>,
    groups: &[usize],
    n_groups: usize,
    n_items: usize,
    n_cat: usize,
    specific_map: &[i32],
    n_specific: usize,
    a_general: &[Vec<f64>],
    a_specific: &[Vec<f64>],
    threshold: &[Vec<f64>],
    general_mean: &[f64],
    general_sd: &[f64],
    specific_mean: &[Vec<f64>],
    specific_sd: &[Vec<f64>],
    q_general: usize,
    q_specific: usize,
) -> Result<OrdinalSrmsr, String> {
    if n_groups < 2 || n_items < 2 || n_cat < 2 || groups.is_empty() || y.len() != groups.len() * n_items {
        return Err("invalid ordinal SRMSR dimensions".into());
    }
    if groups.iter().any(|&group| group >= n_groups) || specific_map.len() != n_items {
        return Err("invalid group or specific map".into());
    }
    if observed.is_some_and(|mask| mask.len() != y.len()) {
        return Err("observed mask length mismatch".into());
    }
    let boundaries = n_cat - 1;
    let widths = [n_items, n_items, n_items * boundaries];
    for (name, rows, width) in [("slope", a_general, widths[0]), ("specific slope", a_specific, widths[1]), ("threshold", threshold, widths[2])] {
        if rows.len() != n_groups || rows.iter().any(|row| row.len() != width) {
            return Err(format!("invalid {name} geometry"));
        }
    }
    if general_mean.len() != n_groups || general_sd.len() != n_groups {
        return Err("invalid general latent moments".into());
    }
    if specific_mean.len() != n_groups || specific_sd.len() != n_groups
        || specific_mean.iter().chain(specific_sd).any(|row| row.len() != n_specific)
    {
        return Err("invalid specific latent moments".into());
    }
    let pairs = n_items * (n_items - 1) / 2;
    let mut residuals = Vec::with_capacity(n_groups * pairs);
    let mut group_rows = Vec::with_capacity(n_groups);
    let mut unweighted = 0.0;
    let mut weighted = 0.0;
    let mut weight_total = 0.0;
    for group in 0..n_groups {
        let (observed_moments, pair_counts) = observed_moments(y, observed, groups, group, n_items, n_cat)?;
        let model = model_moments(
            &a_general[group], &a_specific[group], &threshold[group], specific_map, n_specific,
            general_mean[group], general_sd[group], &specific_mean[group], &specific_sd[group],
            n_items, n_cat, q_general, q_specific,
        )?;
        let mut sum_squared = 0.0;
        for i in 0..n_items {
            for j in i + 1..n_items {
                let idx = pair_index(i, j, n_items);
                let observed_variance_i = observed_moments.second[i] - observed_moments.mean[i].powi(2);
                let observed_variance_j = observed_moments.second[j] - observed_moments.mean[j].powi(2);
                let model_variance_i = model.second[i] - model.mean[i].powi(2);
                let model_variance_j = model.second[j] - model.mean[j].powi(2);
                if observed_variance_i <= 0.0 || observed_variance_j <= 0.0 || model_variance_i <= 0.0 || model_variance_j <= 0.0 {
                    return Err("zero item variance prevents declared ordinal correlation".into());
                }
                let observed_corr = (observed_moments.cross[idx] - observed_moments.mean[i] * observed_moments.mean[j])
                    / (observed_variance_i * observed_variance_j).sqrt();
                let expected_corr = (model.cross[idx] - model.mean[i] * model.mean[j])
                    / (model_variance_i * model_variance_j).sqrt();
                if !observed_corr.is_finite() || !expected_corr.is_finite() {
                    return Err("nonfinite declared ordinal correlation".into());
                }
                let residual = observed_corr - expected_corr;
                sum_squared += residual * residual;
                residuals.push(PairResidual {
                    group, item_i: i, item_j: j, n_pair: pair_counts[idx], observed_corr, expected_corr, residual,
                });
            }
        }
        let srmsr = (sum_squared / pairs as f64).sqrt();
        let n_group = groups.iter().filter(|&&value| value == group).count();
        unweighted += sum_squared;
        weighted += n_group as f64 * sum_squared;
        weight_total += n_group as f64;
        group_rows.push(GroupOrdinalSrmsr { group, n_group, n_pairs: pairs, srmsr });
    }
    Ok(OrdinalSrmsr {
        definition: "ordinal_score_pearson_srmsr",
        category_scores: "0..=n_cat-1",
        group_aggregation: "unweighted_and_group_size_weighted",
        q_general,
        q_specific,
        n_groups,
        n_items,
        n_cat,
        n_specific,
        n_pairs_per_group: pairs,
        n_total_pairs: n_groups * pairs,
        unweighted_srmsr: (unweighted / (n_groups * pairs) as f64).sqrt(),
        group_size_weighted_srmsr: (weighted / (weight_total * pairs as f64)).sqrt(),
        groups: group_rows,
        residuals,
    })
}
