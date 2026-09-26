//! Saved-fit person fit for graded-response bifactor and two-tier models.
//!
//! The reported `l_z` is the observed conditional log likelihood centred and
//! scaled by its model expectation and variance at posterior-mean traits.
//! The statistic is descriptive at estimated traits; no asymptotic `l_z*` or
//! normal-reference p value is claimed for these cross-loading models.
//!
//! References (APA 7th ed.):
//! Albers, C. J., Meijer, R. R., & Tendeiro, J. N. (2016). Derivation and
//! applicability of asymptotic results for multiple subtests person-fit
//! statistics. *Applied Psychological Measurement, 40*(4), 274–288.
//! https://doi.org/10.1177/0146621615622832 (pp. 276–278, Equations 2–4, 9).
//! Cai, L. (2010). A two-tier full-information item factor analysis model with
//! applications. *Psychometrika, 75*, 581–612.
//! https://doi.org/10.1007/s11336-010-9178-0 (model and dimension reduction).

use crate::parallel::{lcg_uniform, normal_draw};
use crate::poly::grm_logprobs;
use crate::two_tier_grm::{cholesky_lower, gh_rule};

/// Group-major item parameters for either saved fit. A bifactor model has
/// `n_primary = 1`; a two-tier model has `n_groups = 1`.
pub struct MultidimPersonFitInput<'a> {
    pub y: &'a [usize],
    pub observed: Option<&'a [bool]>,
    pub group: &'a [usize],
    pub specific_map: &'a [i32],
    pub a_primary: &'a [f64],
    pub a_specific: &'a [f64],
    pub threshold: &'a [f64],
    pub primary_mean: &'a [f64],
    pub primary_cov: &'a [f64],
    pub specific_sd: &'a [f64],
    pub n_persons: usize,
    pub n_items: usize,
    pub n_primary: usize,
    pub n_specific: usize,
    pub n_groups: usize,
    pub n_cat: usize,
    pub q_primary: usize,
    pub q_specific: usize,
    pub flag_threshold: f64,
}

pub struct MultidimPersonFit {
    pub lz: Vec<f64>,
    pub primary_eap: Vec<f64>,
    pub specific_eap: Vec<f64>,
    pub n_observed: Vec<usize>,
    pub flagged: Vec<bool>,
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}

/// Compute posterior-mean primary and specific traits using product
/// Gauss-Hermite integration with specific-block factorization, then evaluate
/// the conditional `l_z` of Albers et al. (2016, pp. 276–278, Equations 2–4).
/// The user supplies both quadrature node counts; neither is capped or
/// defaulted. `l_z` is not assigned a standard-normal null distribution here.
///
/// References (APA 7th ed.): Albers, C. J., Meijer, R. R., & Tendeiro,
/// J. N. (2016). Derivation and applicability of asymptotic results for
/// multiple subtests person-fit statistics. *Applied Psychological
/// Measurement, 40*(4), 274–288. https://doi.org/10.1177/0146621615622832;
/// Cai, L. (2010). A two-tier full-information item factor analysis model with
/// applications. *Psychometrika, 75*, 581–612.
/// https://doi.org/10.1007/s11336-010-9178-0 (pp. 586–587, Equations 1–5).
pub fn person_fit_multidim(
    input: &MultidimPersonFitInput<'_>,
) -> Result<MultidimPersonFit, String> {
    let x = input;
    if x.n_persons == 0
        || x.n_items == 0
        || x.n_primary == 0
        || x.n_specific == 0
        || x.n_groups == 0
        || x.n_cat < 2
        || x.q_primary == 0
        || x.q_specific == 0
        || !x.flag_threshold.is_finite()
    {
        return Err("invalid person-fit dimensions, quadrature count, or threshold".into());
    }
    let cells = x
        .n_persons
        .checked_mul(x.n_items)
        .ok_or("response size overflow")?;
    let gp = x
        .n_groups
        .checked_mul(x.n_primary)
        .ok_or("parameter size overflow")?;
    let gi = x
        .n_groups
        .checked_mul(x.n_items)
        .ok_or("parameter size overflow")?;
    let gpi = gi
        .checked_mul(x.n_primary)
        .ok_or("parameter size overflow")?;
    let gic = gi
        .checked_mul(x.n_cat - 1)
        .ok_or("parameter size overflow")?;
    let gpp = gp
        .checked_mul(x.n_primary)
        .ok_or("parameter size overflow")?;
    let gs = x
        .n_groups
        .checked_mul(x.n_specific)
        .ok_or("parameter size overflow")?;
    if x.y.len() != cells
        || x.observed.is_some_and(|o| o.len() != cells)
        || x.group.len() != x.n_persons
        || x.specific_map.len() != x.n_items
        || x.a_primary.len() != gpi
        || x.a_specific.len() != gi
        || x.threshold.len() != gic
        || x.primary_mean.len() != gp
        || x.primary_cov.len() != gpp
        || x.specific_sd.len() != gs
    {
        return Err("person-fit input shapes do not match".into());
    }
    if x.a_primary
        .iter()
        .chain(x.a_specific)
        .chain(x.threshold)
        .chain(x.primary_mean)
        .chain(x.primary_cov)
        .any(|v| !v.is_finite())
        || x.specific_sd.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("person-fit parameters must be finite with positive specific SD".into());
    }
    for item in 0..x.n_items {
        if x.specific_map[item] < -1
            || usize::try_from(x.specific_map[item]).is_ok_and(|s| s >= x.n_specific)
        {
            return Err("specific_map entry is out of range".into());
        }
        if x.specific_map[item] == -1
            && (0..x.n_groups).any(|g| x.a_specific[g * x.n_items + item] != 0.0)
        {
            return Err("specific-free item has a nonzero specific loading".into());
        }
    }
    for g in 0..x.n_groups {
        let cov =
            &x.primary_cov[g * x.n_primary * x.n_primary..(g + 1) * x.n_primary * x.n_primary];
        for row in 0..x.n_primary {
            for col in 0..row {
                if (cov[row * x.n_primary + col] - cov[col * x.n_primary + row]).abs() > 1e-10 {
                    return Err("primary covariance must be symmetric".into());
                }
            }
        }
        for i in 0..x.n_items {
            let offset = (g * x.n_items + i) * (x.n_cat - 1);
            if x.threshold[offset..offset + x.n_cat - 1]
                .windows(2)
                .any(|pair| pair[0] <= pair[1])
            {
                return Err("GRM thresholds must be strictly decreasing".into());
            }
        }
    }
    for p in 0..x.n_persons {
        if x.group[p] >= x.n_groups {
            return Err("group entry is out of range".into());
        }
        for i in 0..x.n_items {
            let idx = p * x.n_items + i;
            if x.observed.is_none_or(|o| o[idx]) && x.y[idx] >= x.n_cat {
                return Err("observed category is out of range".into());
            }
        }
    }

    let power =
        u32::try_from(x.n_primary).map_err(|_| "primary dimensions exceed grid exponent range")?;
    let n_grid = x
        .q_primary
        .checked_pow(power)
        .ok_or("primary quadrature grid size overflow")?;
    let grid_cells = n_grid
        .checked_mul(x.n_primary)
        .ok_or("grid size overflow")?;
    let primary_cells = x
        .n_persons
        .checked_mul(x.n_primary)
        .ok_or("result size overflow")?;
    let specific_cells = x
        .n_persons
        .checked_mul(x.n_specific)
        .ok_or("result size overflow")?;
    x.n_specific
        .checked_mul(n_grid)
        .ok_or("specific grid size overflow")?;
    let (primary_nodes, primary_weights) = gh_rule(x.q_primary)?;
    let (specific_nodes, specific_weights) = gh_rule(x.q_specific)?;
    let mut standard_coords = Vec::new();
    standard_coords
        .try_reserve_exact(grid_cells)
        .map_err(|_| "primary grid allocation failed")?;
    let mut log_w = Vec::new();
    log_w
        .try_reserve_exact(n_grid)
        .map_err(|_| "primary weight allocation failed")?;
    for node in 0..n_grid {
        let mut tail = node;
        let mut weight = 0.0;
        for _ in 0..x.n_primary {
            let digit = tail % x.q_primary;
            tail /= x.q_primary;
            standard_coords.push(primary_nodes[digit]);
            weight += primary_weights[digit].ln();
        }
        log_w.push(weight);
    }
    let log_ws: Vec<f64> = specific_weights.iter().map(|w| w.ln()).collect();
    let mut coords_by_group = Vec::with_capacity(x.n_groups);
    for group in 0..x.n_groups {
        let cov = &x.primary_cov
            [group * x.n_primary * x.n_primary..(group + 1) * x.n_primary * x.n_primary];
        let (chol, _) = cholesky_lower(cov, x.n_primary)
            .ok_or("primary covariance must be positive definite")?;
        let mut coords = Vec::new();
        coords
            .try_reserve_exact(grid_cells)
            .map_err(|_| "primary grid allocation failed")?;
        for node in 0..n_grid {
            for dim in 0..x.n_primary {
                let value = x.primary_mean[group * x.n_primary + dim]
                    + (0..=dim)
                        .map(|j| {
                            chol[dim * x.n_primary + j] * standard_coords[node * x.n_primary + j]
                        })
                        .sum::<f64>();
                coords.push(value);
            }
        }
        coords_by_group.push(coords);
    }
    let blocks: Vec<Vec<usize>> = (0..x.n_specific)
        .map(|s| {
            (0..x.n_items)
                .filter(|&i| x.specific_map[i] == s as i32)
                .collect()
        })
        .collect();
    let free: Vec<usize> = (0..x.n_items)
        .filter(|&i| x.specific_map[i] == -1)
        .collect();
    let mut result = MultidimPersonFit {
        lz: vec![f64::NAN; x.n_persons],
        primary_eap: vec![0.0; primary_cells],
        specific_eap: vec![0.0; specific_cells],
        n_observed: vec![0; x.n_persons],
        flagged: vec![false; x.n_persons],
    };
    for p in 0..x.n_persons {
        let group = x.group[p];
        let coords = &coords_by_group[group];
        let observed = |i: usize| x.observed.is_none_or(|mask| mask[p * x.n_items + i]);
        result.n_observed[p] = (0..x.n_items).filter(|&i| observed(i)).count();
        let item_log = |i: usize, primary: &[f64], specific: f64| -> Vec<f64> {
            let item = group * x.n_items + i;
            let base = (0..x.n_primary)
                .map(|d| x.a_primary[item * x.n_primary + d] * primary[d])
                .sum::<f64>()
                + x.a_specific[item] * specific;
            let offset = item * (x.n_cat - 1);
            grm_logprobs(base, &x.threshold[offset..offset + x.n_cat - 1])
        };
        let mut log_primary = vec![f64::NEG_INFINITY; n_grid];
        let mut conditional_specific = vec![0.0; x.n_specific * n_grid];
        let mut log_h = vec![f64::NEG_INFINITY; x.q_specific];
        for node in 0..n_grid {
            if !log_w[node].is_finite() {
                continue;
            }
            let primary = &coords[node * x.n_primary..(node + 1) * x.n_primary];
            let mut log_like = log_w[node];
            for &i in &free {
                if observed(i) {
                    log_like += item_log(i, primary, 0.0)[x.y[p * x.n_items + i]];
                }
            }
            for s in 0..x.n_specific {
                for h in 0..x.q_specific {
                    log_h[h] = log_ws[h];
                    if !log_h[h].is_finite() {
                        continue;
                    }
                    let trait_s = x.specific_sd[group * x.n_specific + s] * specific_nodes[h];
                    for &i in &blocks[s] {
                        if observed(i) {
                            log_h[h] += item_log(i, primary, trait_s)[x.y[p * x.n_items + i]];
                        }
                    }
                }
                let block_log = log_sum_exp(&log_h);
                if !block_log.is_finite() {
                    return Err("specific posterior has no finite mass".into());
                }
                conditional_specific[s * n_grid + node] = (0..x.q_specific)
                    .map(|h| {
                        (log_h[h] - block_log).exp()
                            * specific_nodes[h]
                            * x.specific_sd[group * x.n_specific + s]
                    })
                    .sum();
                log_like += block_log;
            }
            log_primary[node] = log_like;
        }
        let normalizer = log_sum_exp(&log_primary);
        if !normalizer.is_finite() {
            return Err("primary posterior has no finite mass".into());
        }
        for node in 0..n_grid {
            let weight = (log_primary[node] - normalizer).exp();
            if weight == 0.0 {
                continue;
            }
            for d in 0..x.n_primary {
                result.primary_eap[p * x.n_primary + d] += weight * coords[node * x.n_primary + d];
            }
            for s in 0..x.n_specific {
                result.specific_eap[p * x.n_specific + s] +=
                    weight * conditional_specific[s * n_grid + node];
            }
        }
        if result.n_observed[p] < 2 {
            continue;
        }
        let primary = &result.primary_eap[p * x.n_primary..(p + 1) * x.n_primary];
        let mut centered = 0.0;
        let mut variance = 0.0;
        for i in 0..x.n_items {
            if !observed(i) {
                continue;
            }
            let specific = if x.specific_map[i] < 0 {
                0.0
            } else {
                result.specific_eap[p * x.n_specific + x.specific_map[i] as usize]
            };
            let logs = item_log(i, primary, specific);
            let mean: f64 = logs.iter().map(|&lp| lp.exp() * lp).sum();
            let second: f64 = logs.iter().map(|&lp| lp.exp() * lp * lp).sum();
            centered += logs[x.y[p * x.n_items + i]] - mean;
            variance += (second - mean * mean).max(0.0);
        }
        if variance > 0.0 && variance.is_finite() {
            result.lz[p] = centered / variance.sqrt();
            result.flagged[p] = result.lz[p] < x.flag_threshold;
        }
    }
    Ok(result)
}

/// Calibrate descriptive `l_z` by generating responses under the saved model,
/// retaining each person's observed-item pattern and group, and re-estimating
/// posterior-mean traits for every replicate. The lower-tail probability uses
/// `(1 + count(simulated <= observed)) / (n_reps + 1)`; it conditions on the
/// saved item parameters, rather than claiming a cross-loading `l_z*` limit.
///
/// Basis: Albers, Meijer, and Tendeiro (2016, pp. 279–281) examine finite-sample
/// simulation where asymptotic person-fit calibration is uncertain. Full APA
/// reference: Albers, C. J., Meijer, R. R., & Tendeiro, J. N. (2016).
/// Derivation and applicability of asymptotic results for multiple subtests
/// person-fit statistics. *Applied Psychological Measurement, 40*(4),
/// 274–288. https://doi.org/10.1177/0146621615622832.
pub fn person_fit_multidim_resampling(
    input: &MultidimPersonFitInput<'_>,
    n_reps: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    if n_reps == 0 || n_reps == usize::MAX {
        return Err("n_reps must be in 1..usize::MAX".into());
    }
    let observed = person_fit_multidim(input)?;
    let x = input;
    let cells = x
        .n_persons
        .checked_mul(x.n_items)
        .ok_or("response size overflow")?;
    let mut simulated_y = vec![0; cells];
    let mut counts = vec![0usize; x.n_persons];
    let mut state = seed.max(1);
    let mut chol = Vec::with_capacity(x.n_groups);
    for g in 0..x.n_groups {
        let start = g * x.n_primary * x.n_primary;
        chol.push(
            cholesky_lower(
                &x.primary_cov[start..start + x.n_primary * x.n_primary],
                x.n_primary,
            )
            .ok_or("primary covariance must be positive definite")?
            .0,
        );
    }
    for _ in 0..n_reps {
        for p in 0..x.n_persons {
            let g = x.group[p];
            let z: Vec<f64> = (0..x.n_primary).map(|_| normal_draw(&mut state)).collect();
            let primary: Vec<f64> = (0..x.n_primary)
                .map(|d| {
                    x.primary_mean[g * x.n_primary + d]
                        + (0..=d)
                            .map(|j| chol[g][d * x.n_primary + j] * z[j])
                            .sum::<f64>()
                })
                .collect();
            let specific: Vec<f64> = (0..x.n_specific)
                .map(|s| x.specific_sd[g * x.n_specific + s] * normal_draw(&mut state))
                .collect();
            for i in 0..x.n_items {
                let idx = p * x.n_items + i;
                if x.observed.is_some_and(|mask| !mask[idx]) {
                    continue;
                }
                let item = g * x.n_items + i;
                let s = usize::try_from(x.specific_map[i]).ok();
                let base = (0..x.n_primary)
                    .map(|d| x.a_primary[item * x.n_primary + d] * primary[d])
                    .sum::<f64>()
                    + s.map_or(0.0, |s| x.a_specific[item] * specific[s]);
                let start = item * (x.n_cat - 1);
                let logs = grm_logprobs(base, &x.threshold[start..start + x.n_cat - 1]);
                let draw = lcg_uniform(&mut state);
                let mut cumulative = 0.0;
                let mut category = x.n_cat - 1;
                for (k, lp) in logs.iter().enumerate() {
                    cumulative += lp.exp();
                    if draw < cumulative {
                        category = k;
                        break;
                    }
                }
                simulated_y[idx] = category;
            }
        }
        let simulated = person_fit_multidim(&MultidimPersonFitInput {
            y: &simulated_y,
            ..*x
        })?;
        for p in 0..x.n_persons {
            if observed.lz[p].is_finite() && !simulated.lz[p].is_finite() {
                return Err("simulated person-fit statistic is not finite".into());
            }
            if observed.lz[p].is_finite() && simulated.lz[p] <= observed.lz[p] {
                counts[p] += 1;
            }
        }
    }
    Ok((0..x.n_persons)
        .map(|p| {
            if observed.lz[p].is_finite() {
                (counts[p] + 1) as f64 / (n_reps + 1) as f64
            } else {
                f64::NAN
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{poly_person_fit, PolyModel};

    #[test]
    fn reduction_to_unidimensional_grm_and_missingness() {
        let y = [0, 2, 1, 2, 0, 1, 1, 0, 2];
        let mask = [true, true, true, true, true, true, true, false, true];
        let slopes = [0.8, 1.2, 0.6];
        let thresholds = [1.0, -1.0, 0.8, -0.8, 1.2, -0.5];
        let input = MultidimPersonFitInput {
            y: &y,
            observed: Some(&mask),
            group: &[0, 0, 0],
            specific_map: &[-1, -1, -1],
            a_primary: &slopes,
            a_specific: &[0.0; 3],
            threshold: &thresholds,
            primary_mean: &[0.0],
            primary_cov: &[1.0],
            specific_sd: &[1.0],
            n_persons: 3,
            n_items: 3,
            n_primary: 1,
            n_specific: 1,
            n_groups: 1,
            n_cat: 3,
            q_primary: 121,
            q_specific: 121,
            flag_threshold: -1.5,
        };
        let got = person_fit_multidim(&input).unwrap();
        let expected = poly_person_fit(
            &y,
            Some(&mask),
            3,
            3,
            3,
            &slopes,
            &thresholds,
            PolyModel::Grm,
            121,
            0.0,
            1.0,
            -1.5,
        )
        .unwrap();
        for p in 0..3 {
            assert!((got.primary_eap[p] - expected.theta_eap[p]).abs() < 1e-10);
            assert!((got.lz[p] - expected.lz[p]).abs() < 1e-10);
        }
        assert_eq!(got.n_observed, [3, 3, 2]);
        assert!(got.specific_eap.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn group_prior_and_specific_block_affect_posterior() {
        let input = MultidimPersonFitInput {
            y: &[2, 2, 2, 2, 2, 2],
            observed: None,
            group: &[0, 1],
            specific_map: &[0, 0, -1],
            a_primary: &[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            a_specific: &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            threshold: &[
                1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            ],
            primary_mean: &[0.0, 1.0],
            primary_cov: &[1.0, 1.0],
            specific_sd: &[1.0, 1.0],
            n_persons: 2,
            n_items: 3,
            n_primary: 1,
            n_specific: 1,
            n_groups: 2,
            n_cat: 3,
            q_primary: 121,
            q_specific: 121,
            flag_threshold: -1.5,
        };
        let got = person_fit_multidim(&input).unwrap();
        assert!(got.primary_eap[1] > got.primary_eap[0]);
        assert!(got.specific_eap[0] > 0.0 && got.specific_eap[1] > 0.0);
        assert!(got.lz.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn resampling_reestimates_traits_and_is_seeded() {
        let input = MultidimPersonFitInput {
            y: &[2, 0, 1],
            observed: None,
            group: &[0],
            specific_map: &[0, 0, -1],
            a_primary: &[0.8; 3],
            a_specific: &[1.0, 1.0, 0.0],
            threshold: &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            primary_mean: &[0.0],
            primary_cov: &[1.0],
            specific_sd: &[1.0],
            n_persons: 1,
            n_items: 3,
            n_primary: 1,
            n_specific: 1,
            n_groups: 1,
            n_cat: 3,
            q_primary: 121,
            q_specific: 121,
            flag_threshold: -1.5,
        };
        let first = person_fit_multidim_resampling(&input, 8, 42).unwrap();
        let second = person_fit_multidim_resampling(&input, 8, 42).unwrap();
        assert_eq!(first, second);
        assert!(first[0] >= 1.0 / 9.0 && first[0] <= 1.0);
        assert!(person_fit_multidim_resampling(&input, 0, 42).is_err());
    }

    #[test]
    fn correlated_two_primary_prior_preserves_marginal_grm_fit() {
        let y = [2, 1, 2];
        let thresholds = [1.0, -1.0, 0.8, -0.8, 1.2, -0.5];
        let input = MultidimPersonFitInput {
            y: &y,
            observed: None,
            group: &[0],
            specific_map: &[-1; 3],
            a_primary: &[0.8, 0.0, 1.2, 0.0, 0.6, 0.0],
            a_specific: &[0.0; 3],
            threshold: &thresholds,
            primary_mean: &[0.0, 0.0],
            primary_cov: &[1.0, 0.5, 0.5, 1.0],
            specific_sd: &[1.0],
            n_persons: 1,
            n_items: 3,
            n_primary: 2,
            n_specific: 1,
            n_groups: 1,
            n_cat: 3,
            q_primary: 121,
            q_specific: 121,
            flag_threshold: -1.5,
        };
        let got = person_fit_multidim(&input).unwrap();
        let expected = poly_person_fit(
            &y,
            None,
            1,
            3,
            3,
            &[0.8, 1.2, 0.6],
            &thresholds,
            PolyModel::Grm,
            121,
            0.0,
            1.0,
            -1.5,
        )
        .unwrap();
        assert!((got.primary_eap[0] - expected.theta_eap[0]).abs() < 1e-10);
        assert!((got.primary_eap[1] - 0.5 * got.primary_eap[0]).abs() < 1e-10);
        assert!((got.lz[0] - expected.lz[0]).abs() < 1e-10);
    }
}
