//! Joint ML observed information for the multigroup bifactor GRM.
//!
//! Oakes (1999, eq. 6, p. 480) gives the observed Hessian as the
//! complete-data Hessian plus the derivative of its score through the E-step.
//! The complete data here are standard-normal quadrature nodes. Focal nodes
//! become theta_G = mu_g + sigma_g x and theta_S = tau_gs z, so their
//! coordinates remain fixed when differentiating Q. Cai et al. (2011, p. 230)
//! identify the reference group by zero means and unit variances and link
//! groups through common items. Gibbons et al. (2007, eqs. 9, 15, pp. 7, 9)
//! give the bifactor predictor and reduced E-step used here.
//!
//! References (APA 7th ed.): Oakes, D. (1999). Direct calculation of the
//! information matrix via the EM algorithm. *Journal of the Royal Statistical
//! Society: Series B, 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221–248.
//! https://doi.org/10.1037/a0023350
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
//! Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
//! A. (2007). Full-information item bifactor analysis of graded response
//! data. *Applied Psychological Measurement, 31*(1), 4–19.
//! https://doi.org/10.1177/0146621606289485

use crate::bifactor_grm::{
    BifactorGrmConfig, ItemParams, Validated, check_param_shapes, e_step_multigroup, gh_rule,
    validate,
};
use crate::bifactor_oakes::{BifactorOakesConfig, BifactorOakesResult, bifactor_oakes_se};
use crate::poly::{grm_node_gradient, grm_node_hessian};

struct Joint<'a> {
    y: &'a [usize],
    observed: Option<&'a [bool]>,
    group: &'a [usize],
    v: Validated,
    n_groups: usize,
    specs: Vec<Vec<Vec<usize>>>,
    x: Vec<f64>,
    z: Vec<f64>,
    log_wg: Vec<f64>,
    log_ws: Vec<f64>,
    estimate_specific_vars: bool,
    pop_start: usize,
}

impl Joint<'_> {
    fn unpack(&self, p: &[f64]) -> (Vec<Vec<ItemParams>>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let mut items = Vec::with_capacity(self.n_groups);
        for g in 0..self.n_groups {
            let mut row = Vec::with_capacity(self.v.n_items);
            for i in 0..self.v.n_items {
                let s = &self.specs[g][i];
                let block = self.v.item_block[i].is_some();
                let off = if block { 2 } else { 1 };
                row.push(ItemParams {
                    a_g: p[s[0]],
                    a_s: block.then(|| p[s[1]]),
                    d: s[off..].iter().map(|&j| p[j]).collect(),
                });
            }
            items.push(row);
        }
        let mut mu = vec![0.0; self.n_groups];
        let mut sd = vec![1.0; self.n_groups];
        let mut tau = vec![vec![1.0; self.v.n_specific]; self.n_groups];
        let mut j = self.pop_start;
        for g in 1..self.n_groups {
            mu[g] = p[j];
            sd[g] = p[j + 1].sqrt();
            j += 2;
            if self.estimate_specific_vars {
                for s in 0..self.v.n_specific {
                    tau[g][s] = p[j].sqrt();
                    j += 1;
                }
            }
        }
        (items, mu, sd, tau)
    }

    fn posterior(&self, p: &[f64]) -> Vec<Vec<Vec<Vec<f64>>>> {
        let (items, mu, sd, tau) = self.unpack(p);
        let tg: Vec<Vec<f64>> = (0..self.n_groups)
            .map(|g| self.x.iter().map(|&x| mu[g] + sd[g] * x).collect())
            .collect();
        let ts: Vec<Vec<Vec<f64>>> = (0..self.n_groups)
            .map(|g| {
                (0..self.v.n_specific)
                    .map(|s| self.z.iter().map(|&z| tau[g][s] * z).collect())
                    .collect()
            })
            .collect();
        let (_, counts, _, _, _, _, _) = e_step_multigroup(
            &self.v,
            self.y,
            self.observed,
            self.group,
            self.n_groups,
            &items,
            &tg,
            &ts,
            &self.log_wg,
            &self.log_ws,
            self.x.len(),
            self.z.len(),
            crate::Device::Cpu,
        );
        counts
    }

    // Returns analytic Q score and, when requested, its fixed-posterior Hessian.
    fn derivatives(
        &self,
        p: &[f64],
        counts: &[Vec<Vec<Vec<f64>>>],
        hessian: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let k = p.len();
        let mut grad = vec![0.0; k];
        let mut hess = if hessian {
            vec![0.0; k * k]
        } else {
            Vec::new()
        };
        let (_, mu, sd, tau) = self.unpack(p);
        for g in 0..self.n_groups {
            let pop = self.pop_start
                + (g.saturating_sub(1))
                    * (2 + usize::from(self.estimate_specific_vars) * self.v.n_specific);
            for i in 0..self.v.n_items {
                let slots = &self.specs[g][i];
                let block = self.v.item_block[i];
                let off = if block.is_some() { 2 } else { 1 };
                let ag = p[slots[0]];
                let as_ = block.map_or(0.0, |_| p[slots[1]]);
                let d: Vec<f64> = slots[off..].iter().map(|&j| p[j]).collect();
                for (node, cell) in counts[g][i].iter().enumerate() {
                    let t = if block.is_some() {
                        node / self.z.len()
                    } else {
                        node
                    };
                    let h = if block.is_some() {
                        node % self.z.len()
                    } else {
                        0
                    };
                    let x = self.x[t];
                    let z = if block.is_some() { self.z[h] } else { 0.0 };
                    let theta_g = mu[g] + sd[g] * x;
                    let theta_s = block.map_or(0.0, |s| tau[g][s] * z);
                    let base = ag * theta_g + as_ * theta_s;
                    let (gb, gd) = grm_node_gradient(base, &d, cell);
                    let mut jac = vec![(slots[0], theta_g)];
                    if block.is_some() {
                        jac.push((slots[1], theta_s));
                    }
                    if g > 0 {
                        jac.push((pop, ag));
                        jac.push((pop + 1, ag * x / (2.0 * sd[g])));
                        if let Some(s) = block {
                            if self.estimate_specific_vars {
                                jac.push((pop + 2 + s, as_ * z / (2.0 * tau[g][s])));
                            }
                        }
                    }
                    for &(j, value) in &jac {
                        grad[j] += gb * value;
                    }
                    for (r, &value) in gd.iter().enumerate() {
                        grad[slots[off + r]] += value;
                    }
                    if !hessian {
                        continue;
                    }
                    let (grand, rows, mat) = grm_node_hessian(base, &d, cell);
                    for &(j, a) in &jac {
                        for &(l, b) in &jac {
                            hess[j * k + l] += grand * a * b;
                        }
                        for (r, &v) in rows.iter().enumerate() {
                            let l = slots[off + r];
                            hess[j * k + l] += a * v;
                            hess[l * k + j] += a * v;
                        }
                    }
                    if g > 0 {
                        for (j, l, value) in
                            [(slots[0], pop, 1.0), (slots[0], pop + 1, x / (2.0 * sd[g]))]
                        {
                            hess[j * k + l] += gb * value;
                            hess[l * k + j] += gb * value;
                        }
                        if let Some(s) = block {
                            if self.estimate_specific_vars {
                                let l = pop + 2 + s;
                                hess[slots[1] * k + l] += gb * z / (2.0 * tau[g][s]);
                                hess[l * k + slots[1]] += gb * z / (2.0 * tau[g][s]);
                                hess[l * k + l] -= gb * as_ * z / (4.0 * tau[g][s].powi(3));
                            }
                        }
                        hess[(pop + 1) * k + pop + 1] -= gb * ag * x / (4.0 * sd[g].powi(3));
                    }
                    for (r, &jr) in slots[off..].iter().enumerate() {
                        for (c, &jc) in slots[off..].iter().enumerate() {
                            hess[jr * k + jc] += mat[r][c];
                        }
                    }
                }
            }
        }
        (grad, hess)
    }
}

/// Joint ML Oakes information in label order: anchored items by item,
/// free items by item then group, each as `a_general, a_specific?, d:0..`;
/// then each focal group's `general_mean, general_var, specific_var:0..`.
/// The reference group's distribution is fixed. Node counts and FD step are
/// caller-owned. Basis: Oakes (1999, eq. 6, p. 480); Cai et al. (2011, p. 230);
/// Gibbons et al. (2007, eqs. 9, 15, pp. 7, 9). Full APA references above.
#[allow(clippy::too_many_arguments)]
pub fn bifactor_multigroup_oakes_se(
    a_general: &[Vec<f64>],
    a_specific: &[Vec<f64>],
    threshold: &[Vec<f64>],
    general_mean: &[f64],
    general_sd: &[f64],
    specific_sd: &[Vec<f64>],
    y: &[usize],
    observed: Option<&[bool]>,
    group: &[usize],
    n_groups: usize,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    anchor: &[bool],
    estimate_specific_vars: bool,
    cfg: &BifactorOakesConfig,
) -> Result<BifactorOakesResult, String> {
    if n_groups == 0 {
        return Err("n_groups must be >= 1".into());
    }
    if group.len() != n_persons || group.iter().any(|&g| g >= n_groups) {
        return Err("group must have length n_persons and labels in 0..n_groups".into());
    }
    if anchor.len() != n_items {
        return Err("anchor must have length n_items".into());
    }
    if !cfg.fd_step.is_finite() || cfg.fd_step <= 0.0 {
        return Err("fd_step must be finite and positive".into());
    }
    if a_general.len() != n_groups
        || a_specific.len() != n_groups
        || threshold.len() != n_groups
        || general_mean.len() != n_groups
        || general_sd.len() != n_groups
        || specific_sd.len() != n_groups
    {
        return Err("parameter tables must have n_groups rows".into());
    }
    if general_mean[0] != 0.0
        || general_sd[0] != 1.0
        || specific_sd.iter().any(|row| row.len() != n_specific)
        || specific_sd[0].iter().any(|&v| v != 1.0)
    {
        return Err("reference distribution must have zero mean and unit SDs".into());
    }
    for g in 0..n_groups {
        if !general_mean[g].is_finite()
            || !general_sd[g].is_finite()
            || general_sd[g] <= 0.0
            || specific_sd[g].iter().any(|&v| !v.is_finite() || v <= 0.0)
        {
            return Err("group means and SDs must be finite; SDs positive".into());
        }
        if !general_sd[g].powi(2).is_finite()
            || specific_sd[g].iter().any(|&v| !v.powi(2).is_finite())
        {
            return Err("group variances must be finite".into());
        }
        if !estimate_specific_vars && specific_sd[g].iter().any(|&v| v != 1.0) {
            return Err("specific_sd must be one unless estimate_specific_vars is true".into());
        }
    }
    if n_groups == 1 {
        return bifactor_oakes_se(
            &a_general[0],
            &a_specific[0],
            &threshold[0],
            y,
            observed,
            specific_map,
            n_persons,
            n_items,
            n_specific,
            n_cat,
            cfg,
        );
    }
    if !anchor.iter().any(|&a| a) {
        return Err("at least one anchored item is required".into());
    }
    let mut group_n = vec![0; n_groups];
    for &g in group {
        group_n[g] += 1;
    }
    if group_n.contains(&0) {
        return Err("every group must contain a person".into());
    }
    let config = BifactorGrmConfig {
        q_general: cfg.q_general,
        q_specific: cfg.q_specific,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1e-8,
        device: crate::Device::Cpu,
    };
    let v = validate(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &config,
    )?;
    n_groups
        .checked_mul(n_items)
        .and_then(|v| v.checked_mul(cfg.q_general))
        .and_then(|v| v.checked_mul(cfg.q_specific))
        .and_then(|v| v.checked_mul(n_cat))
        .ok_or_else(|| "multigroup quadrature working set overflows usize".to_string())?;
    for i in 0..n_items {
        if !anchor[i] {
            for g in 0..n_groups {
                let mut seen = vec![false; n_cat];
                for p in 0..n_persons {
                    if group[p] == g && observed.is_none_or(|o| o[p * n_items + i]) {
                        seen[y[p * n_items + i]] = true;
                    }
                }
                if seen.contains(&false) {
                    return Err(format!(
                        "free item {i} has unobserved category in group {g}"
                    ));
                }
            }
        }
    }
    for g in 0..n_groups {
        check_param_shapes(&v, &a_general[g], &a_specific[g], &threshold[g])?;
        for i in 0..n_items {
            if anchor[i]
                && (a_general[g][i] != a_general[0][i]
                    || a_specific[g][i] != a_specific[0][i]
                    || threshold[g][i * v.m1..(i + 1) * v.m1]
                        != threshold[0][i * v.m1..(i + 1) * v.m1])
            {
                return Err(format!("anchored item {i} differs in group {g}"));
            }
        }
    }
    let mut specs = vec![vec![Vec::new(); n_items]; n_groups];
    let mut labels = Vec::new();
    let mut packed = Vec::new();
    for i in 0..n_items {
        for g in 0..n_groups {
            if anchor[i] && g > 0 {
                specs[g][i] = specs[0][i].clone();
                continue;
            }
            let suffix = if anchor[i] {
                format!(":{i}")
            } else {
                format!(":{g}:{i}")
            };
            let mut slots = Vec::new();
            slots.push(packed.len());
            packed.push(a_general[g][i]);
            labels.push(format!("a_general{suffix}"));
            if v.item_block[i].is_some() {
                slots.push(packed.len());
                packed.push(a_specific[g][i]);
                labels.push(format!("a_specific{suffix}"));
            }
            for t in 0..v.m1 {
                slots.push(packed.len());
                packed.push(threshold[g][i * v.m1 + t]);
                labels.push(if anchor[i] {
                    format!("d:{i}:{t}")
                } else {
                    format!("d:{g}:{i}:{t}")
                });
            }
            specs[g][i] = slots;
        }
    }
    let pop_start = packed.len();
    for g in 1..n_groups {
        labels.push(format!("general_mean:{g}"));
        packed.push(general_mean[g]);
        labels.push(format!("general_var:{g}"));
        packed.push(general_sd[g].powi(2));
        if estimate_specific_vars {
            for s in 0..n_specific {
                labels.push(format!("specific_var:{g}:{s}"));
                packed.push(specific_sd[g][s].powi(2));
            }
        }
    }
    let (x, wg) = gh_rule(cfg.q_general)?;
    let (z, ws) = gh_rule(cfg.q_specific)?;
    let joint = Joint {
        y,
        observed,
        group,
        v,
        n_groups,
        specs,
        x: x.to_vec(),
        z: z.to_vec(),
        log_wg: wg.iter().map(|w| w.ln()).collect(),
        log_ws: ws.iter().map(|w| w.ln()).collect(),
        estimate_specific_vars,
        pop_start,
    };
    let posterior = joint.posterior(&packed);
    let (g0, a) = joint.derivatives(&packed, &posterior, true);
    let k = packed.len();
    let mut cross = vec![0.0; k * k];
    for j in 0..k {
        let step = cfg.fd_step * (1.0 + packed[j].abs());
        let mut perturbed = packed.clone();
        perturbed[j] += step;
        if j >= pop_start && packed[j] > 0.0 && perturbed[j] <= 0.0 {
            perturbed[j] = packed[j] - step;
        }
        let posterior_j = joint.posterior(&perturbed);
        let (gj, _) = joint.derivatives(&packed, &posterior_j, false);
        for c in 0..k {
            cross[j * k + c] = (gj[c] - g0[c]) / (perturbed[j] - packed[j]);
        }
    }
    let mut information = vec![0.0; k * k];
    for r in 0..k {
        for c in 0..k {
            information[r * k + c] =
                -0.5 * (a[r * k + c] + cross[r * k + c] + a[c * k + r] + cross[c * k + r]);
        }
    }
    super::bifactor_oakes::finish_information(labels, information)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_group_is_exact_and_rejects_bad_inputs() {
        let y = [0, 0, 1, 1, 0, 1, 1, 0];
        let smap = [0, 0];
        let ag = vec![0.8, 1.0];
        let as_ = vec![0.5, 0.7];
        let th = vec![0.1, -0.2];
        let cfg = BifactorOakesConfig {
            q_general: 3,
            q_specific: 3,
            fd_step: 1e-5,
        };
        let single = bifactor_oakes_se(&ag, &as_, &th, &y, None, &smap, 4, 2, 1, 2, &cfg).unwrap();
        let multi = bifactor_multigroup_oakes_se(
            &[ag],
            &[as_],
            &[th],
            &[0.0],
            &[1.0],
            &[vec![1.0]],
            &y,
            None,
            &[0; 4],
            1,
            &smap,
            4,
            2,
            1,
            2,
            &[true; 2],
            false,
            &cfg,
        )
        .unwrap();
        assert_eq!(single.labels, multi.labels);
        assert_eq!(single.information, multi.information);
        assert_eq!(single.vcov, multi.vcov);
        assert_eq!(single.se, multi.se);
        assert!(
            bifactor_multigroup_oakes_se(
                &[vec![0.8, 1.0]],
                &[vec![0.5, 0.7]],
                &[vec![0.1, -0.2]],
                &[0.0],
                &[1.0],
                &[vec![1.0]],
                &y,
                None,
                &[1; 4],
                1,
                &smap,
                4,
                2,
                1,
                2,
                &[true; 2],
                false,
                &cfg,
            )
            .is_err()
        );
    }

    #[test]
    fn joint_labels_and_observed_score_fd() {
        let y = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1];
        let group = [0, 0, 0, 0, 1, 1, 1, 1];
        let smap = [0, 0];
        let ag = [vec![0.8, 1.0], vec![0.8, 1.1]];
        let as_ = [vec![0.5, 0.7], vec![0.5, 0.8]];
        let th = [vec![0.1, -0.2], vec![0.1, -0.1]];
        let cfg = BifactorOakesConfig {
            q_general: 5,
            q_specific: 5,
            fd_step: 1e-6,
        };
        let result = bifactor_multigroup_oakes_se(
            &ag,
            &as_,
            &th,
            &[0.0, 0.2],
            &[1.0, 1.1],
            &[vec![1.0], vec![1.2]],
            &y,
            None,
            &group,
            2,
            &smap,
            8,
            2,
            1,
            2,
            &[true, false],
            true,
            &cfg,
        )
        .unwrap();
        let mut bad_ag = ag.clone();
        bad_ag[1][0] = 0.9;
        assert!(
            bifactor_multigroup_oakes_se(
                &bad_ag,
                &as_,
                &th,
                &[0.0, 0.2],
                &[1.0, 1.1],
                &[vec![1.0], vec![1.2]],
                &y,
                None,
                &group,
                2,
                &smap,
                8,
                2,
                1,
                2,
                &[true, false],
                true,
                &cfg,
            )
            .unwrap_err()
            .contains("anchored item")
        );
        assert_eq!(
            result.labels,
            [
                "a_general:0",
                "a_specific:0",
                "d:0:0",
                "a_general:0:1",
                "a_specific:0:1",
                "d:0:1:0",
                "a_general:1:1",
                "a_specific:1:1",
                "d:1:1:0",
                "general_mean:1",
                "general_var:1",
                "specific_var:1:0",
            ]
        );
        let k = result.labels.len();
        assert_eq!(result.information.len(), k * k);
        assert!(result.information.iter().all(|v| v.is_finite()));
        let v = validate(
            &y,
            None,
            &smap,
            8,
            2,
            1,
            2,
            &BifactorGrmConfig {
                q_general: 5,
                q_specific: 5,
                max_iter: 1,
                tol: 1.0,
                n_starts: 1,
                seed: 0,
                newton_iter: 1,
                ridge: 1e-8,
                device: crate::Device::Cpu,
            },
        )
        .unwrap();
        let (x, wg) = gh_rule(5).unwrap();
        let (z, ws) = gh_rule(5).unwrap();
        let joint = Joint {
            y: &y,
            observed: None,
            group: &group,
            v,
            n_groups: 2,
            specs: vec![
                vec![vec![0, 1, 2], vec![3, 4, 5]],
                vec![vec![0, 1, 2], vec![6, 7, 8]],
            ],
            x: x.to_vec(),
            z: z.to_vec(),
            log_wg: wg.iter().map(|w| w.ln()).collect(),
            log_ws: ws.iter().map(|w| w.ln()).collect(),
            estimate_specific_vars: true,
            pop_start: 9,
        };
        let p = [
            0.8, 0.5, 0.1, 1.0, 0.7, -0.2, 1.1, 0.8, -0.1, 0.2, 1.21, 1.44,
        ];
        let mut error_sq = 0.0;
        let mut reference_sq = 0.0;
        for j in 0..k {
            let mut plus = p;
            let mut minus = p;
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let score_plus = joint.derivatives(&plus, &joint.posterior(&plus), false).0;
            let score_minus = joint.derivatives(&minus, &joint.posterior(&minus), false).0;
            for c in 0..k {
                let fd = -(score_plus[c] - score_minus[c]) / (2e-5);
                let observed = result.information[j * k + c];
                let rel = (fd - observed).abs() / (1.0 + fd.abs());
                assert!(
                    rel < 1e-5,
                    "coordinate {j},{c}: {fd} vs {observed}; rel={rel}"
                );
                error_sq += (fd - observed).powi(2);
                reference_sq += fd.powi(2);
            }
        }
        assert!((error_sq / reference_sq).sqrt() < 1e-5);
    }
}
