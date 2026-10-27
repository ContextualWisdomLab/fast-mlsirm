//! Documentary typed execution evidence. Validation proves structure/binding,
//! not authenticated execution. Missing legacy fields are incomplete evidence.
use super::*;
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NativeStageReceipt {
    pub objective: usize,
    pub gradient: usize,
    pub hessian: usize,
    pub prior: usize,
    pub solve: usize,
    pub line_search: usize,
    pub parameter_update: usize,
    pub e_step_gpu_dispatches: usize,
    pub m_step_gpu_dispatches: usize,
    pub cpu_numerical_fallbacks: usize,
    pub iteration_limit_updates: usize,
}
#[cfg(all(feature = "gpu", not(coverage)))]
impl From<crate::gpu_mstep::StageReceipt> for NativeStageReceipt {
    fn from(r: crate::gpu_mstep::StageReceipt) -> Self {
        Self {
            objective: r.objective,
            gradient: r.gradient,
            hessian: r.hessian,
            prior: r.prior,
            solve: r.solve,
            line_search: r.line_search,
            parameter_update: r.parameter_update,
            e_step_gpu_dispatches: r.e_step_gpu_dispatches,
            m_step_gpu_dispatches: r.m_step_gpu_dispatches,
            cpu_numerical_fallbacks: r.cpu_numerical_fallbacks,
            iteration_limit_updates: r.iteration_limit_updates,
        }
    }
}
impl NativeStageReceipt {
    pub(super) fn validate(&self) -> bool {
        self.cpu_numerical_fallbacks == 0
            && self.e_step_gpu_dispatches == 0
            && self.iteration_limit_updates <= 1
            && self.gradient > 0
            && self.hessian > 0
            && self.objective >= self.gradient
            && self.prior == self.objective
            && self.hessian <= self.gradient
            && self.solve <= self.hessian
            && self.parameter_update <= self.solve
            && self.line_search >= self.parameter_update
            && self.m_step_gpu_dispatches == usize::from(self.parameter_update > 0)
    }
}
/// Coordinates copied at native call boundaries, before final reflection.
/// Thresholds are native intercepts (not log gaps); SDs are not log SDs.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ItemCoordinateReceipt {
    pub a_general: f64,
    pub a_specific: Option<f64>,
    pub threshold: Vec<f64>,
}
impl ItemCoordinateReceipt {
    pub(super) fn from_packed(p: &[f64], specific: bool) -> Self {
        Self {
            a_general: p[0],
            a_specific: specific.then(|| p[1]),
            threshold: p[1 + usize::from(specific)..].to_vec(),
        }
    }
}
/// Dense group/item order is the actual fitter order. No person arrays.
/// E-step, group-CM input and final posterior snapshots use this same raw
/// pre-canonicalization representation. Equality is exact numeric equality
/// (including equivalent signed zero), never an estimated tolerance.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConsumedStateReceipt {
    pub items: Vec<Vec<ItemCoordinateReceipt>>,
    pub general_mean: Vec<f64>,
    pub general_sd: Vec<f64>,
    pub specific_mean: Vec<Vec<f64>>,
    pub specific_sd: Vec<Vec<f64>>,
}
impl ConsumedStateReceipt {
    pub(super) fn capture(
        params: &[Vec<ItemParams>],
        mus: &[f64],
        sigmas: &[f64],
        specific_mus: &[Vec<f64>],
        taus: &[Vec<f64>],
    ) -> Self {
        Self {
            items: params
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|p| ItemCoordinateReceipt {
                            a_general: p.a_g,
                            a_specific: p.a_s,
                            threshold: p.d.clone(),
                        })
                        .collect()
                })
                .collect(),
            general_mean: mus.to_vec(),
            general_sd: sigmas.to_vec(),
            specific_mean: specific_mus.to_vec(),
            specific_sd: taus.to_vec(),
        }
    }
    fn valid(
        &self,
        ng: usize,
        map: &[i32],
        ns: usize,
        m1: usize,
        anchor: &[bool],
        cfg: &BifactorMultigroupConfig,
    ) -> bool {
        if self.items.len() != ng
            || self.general_mean.len() != ng
            || self.general_sd.len() != ng
            || self.specific_mean.len() != ng
            || self.specific_sd.len() != ng
        {
            return false;
        }
        for g in 0..ng {
            if self.items[g].len() != map.len()
                || self.specific_mean[g].len() != ns
                || self.specific_sd[g].len() != ns
                || !self.general_mean[g].is_finite()
                || !self.general_sd[g].is_finite()
                || self.general_sd[g] <= 0.0
                || (g == 0 && (self.general_mean[g] != 0.0 || self.general_sd[g] != 1.0))
            {
                return false;
            }
            for s in 0..ns {
                let mean = self.specific_mean[g][s];
                let sd = self.specific_sd[g][s];
                if !mean.is_finite()
                    || !sd.is_finite()
                    || sd <= 0.0
                    || ((g == 0 || !cfg.estimate_specific_means) && mean != 0.0)
                    || ((g == 0 || !cfg.estimate_specific_vars) && sd != 1.0)
                {
                    return false;
                }
            }
            for (i, p) in self.items[g].iter().enumerate() {
                if !p.a_general.is_finite()
                    || p.a_specific.is_some() != (map[i] >= 0)
                    || p.a_specific.is_some_and(|a| !a.is_finite())
                    || p.threshold.len() != m1
                    || p.threshold.iter().any(|x| !x.is_finite())
                    || (g > 0 && anchor[i] && p != &self.items[0][i])
                {
                    return false;
                }
            }
        }
        true
    }
    fn set_coordinate(&mut self, c: &GroupCoordinateReceipt) {
        match (c.specific, c.is_sd) {
            (None, false) => self.general_mean[c.group] = c.value,
            (None, true) => self.general_sd[c.group] = c.value,
            (Some(s), false) => self.specific_mean[c.group][s] = c.value,
            (Some(s), true) => self.specific_sd[c.group][s] = c.value,
        }
    }
    fn matches_returned(
        &self,
        r: &BifactorMultigroupResult,
        map: &[i32],
        ns: usize,
        anchor: &[bool],
    ) -> bool {
        let ng = self.items.len();
        let ni = map.len();
        let mut ag: Vec<Vec<f64>> = self
            .items
            .iter()
            .map(|row| row.iter().map(|p| p.a_general).collect())
            .collect();
        let mut aspec: Vec<Vec<f64>> = self
            .items
            .iter()
            .map(|row| row.iter().map(|p| p.a_specific.unwrap_or(0.0)).collect())
            .collect();
        let threshold: Vec<Vec<f64>> = self
            .items
            .iter()
            .map(|row| {
                row.iter()
                    .flat_map(|p| p.threshold.iter().copied())
                    .collect()
            })
            .collect();
        let mut mu = self.general_mean.clone();
        let mut sm = self.specific_mean.clone();
        // Same selection order and reflection as the native returned-bank code.
        let Some((g, i)) = (0..ni)
            .filter(|&i| anchor[i])
            .flat_map(|i| (0..ng).map(move |g| (g, i)))
            .max_by(|&(g, i), &(h, j)| ag[g][i].abs().total_cmp(&ag[h][j].abs()))
        else {
            return false;
        };
        if ag[g][i] < 0.0 {
            for row in &mut ag {
                for a in row {
                    *a = -*a;
                }
            }
            for m in &mut mu {
                *m = -*m;
            }
        }
        let blocks: Vec<Vec<usize>> = (0..ns)
            .map(|s| (0..ni).filter(|&i| map[i] == s as i32).collect())
            .collect();
        if blocks.iter().any(Vec::is_empty) {
            return false;
        }
        canonicalize_multigroup_specific(&blocks, anchor, &mut aspec, &mut sm);
        ag == r.a_general
            && aspec == r.a_specific
            && threshold == r.threshold
            && mu == r.general_mean
            && self.general_sd == r.general_sd
            && sm == r.specific_mean
            && self.specific_sd == r.specific_sd
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ItemStageReceipt {
    pub start: usize,
    pub iteration: usize,
    pub item: usize,
    /// None is pooled-anchor scope; Some(0) is free reference scope.
    pub group_scope: Option<usize>,
    pub stages: NativeStageReceipt,
    #[serde(default)]
    pub input: Option<ItemCoordinateReceipt>,
    #[serde(default)]
    pub output: Option<ItemCoordinateReceipt>,
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupCoordinateReceipt {
    pub group: usize,
    pub specific: Option<usize>,
    pub is_sd: bool,
    pub owner: String,
    pub value: f64,
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupStageReceipt {
    pub start: usize,
    pub iteration: usize,
    pub completed: bool,
    pub iteration_limited: bool,
    pub algorithm: String,
    #[serde(default)]
    pub input: Option<ConsumedStateReceipt>,
    pub coordinates: Vec<GroupCoordinateReceipt>,
    pub backend: String,
    pub device_name: String,
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FinalPosteriorReceipt {
    pub start: usize,
    pub iteration: usize,
    pub n_persons: usize,
    pub n_items: usize,
    pub n_specific: usize,
    pub q_general: usize,
    pub q_specific: usize,
    pub gpu_used: bool,
    pub backend: Option<String>,
    pub device_name: Option<String>,
    pub fallback_reason: Option<String>,
    pub output_consumed: bool,
    #[serde(default)]
    pub input: Option<ConsumedStateReceipt>,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecutionEvidence {
    pub schema_version: usize,
    pub device: String,
    pub configured_starts: usize,
    pub embedded: bool,
    pub request_sha256: String,
    pub result_sha256: String,
    pub final_posterior: Vec<FinalPosteriorReceipt>,
    pub host_probability_tables: bool,
    pub host_eap: bool,
    pub host_information: bool,
    pub full_gpu: bool,
}
pub(super) fn request_binding(
    y: &[usize],
    obs: Option<&[bool]>,
    gid: &[usize],
    ng: usize,
    map: &[i32],
    anchor: &[bool],
    cfg: &BifactorMultigroupConfig,
) -> Result<String, String> {
    hash(&(y, obs, gid, ng, map, anchor, format!("{cfg:?}")))
}
fn hash(v: &impl serde::Serialize) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(v).map_err(|e| e.to_string())?)
    ))
}
impl BifactorMultigroupResult {
    pub(super) fn evidence_result_binding(&self) -> Result<String, String> {
        hash(&(
            (
                &self.a_general,
                &self.a_specific,
                &self.threshold,
                &self.general_mean,
                &self.general_sd,
                &self.specific_mean,
                &self.specific_sd,
                &self.loglik_trace,
                self.best_start,
                self.n_iter,
            ),
            (
                self.converged,
                &self.termination_reason,
                &self.theta_g_eap,
                &self.theta_g_sd,
                &self.em_objective_trace,
                &self.group_category_counts,
                self.n_parameters,
                self.final_loglik_change,
                format!("{:?}", self.slope_prior),
            ),
            &self.dispatch,
            &self.start_outcomes,
            self.m_step_gpu_dispatches,
            self.scoring_posterior_gpu,
            self.dispatch_complete,
            self.execution_evidence.as_ref().map(|e| {
                (
                    e.schema_version,
                    &e.device,
                    e.configured_starts,
                    e.embedded,
                    &e.request_sha256,
                    &e.final_posterior,
                    e.host_probability_tables,
                    e.host_eap,
                    e.host_information,
                    e.full_gpu,
                )
            }),
        ))
    }
    pub fn validate_gpu_evidence(
        &self,
        y: &[usize],
        obs: Option<&[bool]>,
        gid: &[usize],
        ng: usize,
        map: &[i32],
        anchor: &[bool],
        cfg: &BifactorMultigroupConfig,
        embedded: bool,
    ) -> Result<(), String> {
        let bad = || {
            "gpu_receipt_required: incomplete or mismatched native execution evidence".to_string()
        };
        let e = self.execution_evidence.as_ref().ok_or_else(bad)?;
        let ni = anchor.len();
        let ns = map
            .iter()
            .copied()
            .filter(|s| *s >= 0)
            .max()
            .map_or(0, |s| s as usize + 1);
        if e.schema_version != 2
            || e.device != "Gpu"
            || cfg.device != crate::Device::Gpu
            || e.configured_starts != cfg.n_starts
            || e.embedded != embedded
            || e.request_sha256 != request_binding(y, obs, gid, ng, map, anchor, cfg)?
            || e.result_sha256 != self.evidence_result_binding()?
            || e.host_probability_tables
            || e.host_eap
            || !e.host_information
            || e.full_gpu
            || ng < 2
            || map.len() != ni
            || !self.scoring_posterior_gpu
            || !self.dispatch_complete
        {
            return Err(bad());
        }
        if ni == 0
            || map.iter().any(|&s| s < -1)
            || self.threshold.len() != ng
            || self.threshold[0].len() % ni != 0
        {
            return Err(bad());
        }
        let m1 = self.threshold[0].len() / ni;
        if m1 == 0 {
            return Err(bad());
        }
        let mut winning_terminal = None;
        let nstart = cfg
            .n_starts
            .checked_add(usize::from(embedded))
            .ok_or_else(bad)?;
        if self.start_outcomes.len() != nstart || self.best_start >= nstart {
            return Err(bad());
        }
        let mut sweep_cursor = 0usize;
        let mut helper_count = 0usize;
        for (start, row) in self.start_outcomes.iter().enumerate() {
            let k = row["n_iter"]
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(bad)?;
            let trace = row["loglik_trace"].as_array().ok_or_else(bad)?;
            let term = row["termination_reason"].as_str().ok_or_else(bad)?;
            let converged = row["converged"].as_bool().ok_or_else(bad)?;
            if row["start"].as_u64() != Some(start as u64)
                || row["embedded"].as_bool() != Some(embedded && start == cfg.n_starts)
                || !row["error"].is_null()
                || k > cfg.max_iter
                || trace.len() != k.checked_add(1).ok_or_else(bad)?
                || trace
                    .iter()
                    .any(|v| v.as_f64().is_none_or(|v| !v.is_finite()))
                || !matches!(
                    term,
                    "tolerance_met" | "max_iter_reached" | "numerical_em_stall"
                )
                || converged != (term == "tolerance_met")
                || (term == "max_iter_reached" && k != cfg.max_iter)
            {
                return Err(bad());
            }
            if start == self.best_start
                && (k != self.n_iter
                    || converged != self.converged
                    || term != self.termination_reason
                    || serde_json::to_value(&self.loglik_trace).map_err(|e| e.to_string())?
                        != row["loglik_trace"])
            {
                return Err(bad());
            }
            for iteration in 0..=k {
                let sweep = self.dispatch.get(sweep_cursor).ok_or_else(bad)?;
                sweep_cursor += 1;
                if sweep.start != start
                    || sweep.iteration != iteration
                    || !sweep.gpu_used
                    || sweep.backend.as_deref().is_none_or(str::is_empty)
                    || sweep.device_name.as_deref().is_none_or(str::is_empty)
                    || sweep.fallback_reason.is_some()
                {
                    return Err(bad());
                }
                let input = sweep.input.as_ref().ok_or_else(bad)?;
                if !input.valid(ng, map, ns, m1, anchor, cfg) {
                    return Err(bad());
                }
                if iteration == k {
                    if !sweep.items.is_empty() || sweep.group_update.is_some() {
                        return Err(bad());
                    }
                    if start == self.best_start {
                        winning_terminal = Some(input);
                    }
                    continue;
                }
                let mut after_items = input.clone();
                let mut cursor = 0usize;
                for (item, &pooled) in anchor.iter().enumerate() {
                    for scope in if pooled {
                        vec![None]
                    } else {
                        (0..ng).map(Some).collect()
                    } {
                        let r = sweep.items.get(cursor).ok_or_else(bad)?;
                        cursor += 1;
                        helper_count = helper_count.checked_add(1).ok_or_else(bad)?;
                        if r.start != start
                            || r.iteration != iteration
                            || r.item != item
                            || r.group_scope != scope
                            || !r.stages.validate()
                        {
                            return Err(bad());
                        }
                        if r.input.as_ref() != Some(&input.items[scope.unwrap_or(0)][item]) {
                            return Err(bad());
                        }
                        let output = r.output.as_ref().ok_or_else(bad)?;
                        if let Some(g) = scope {
                            after_items.items[g][item] = output.clone();
                        } else {
                            for row in &mut after_items.items {
                                row[item] = output.clone();
                            }
                        }
                        if iteration + 1 == k && converged && r.stages.iteration_limit_updates != 0
                        {
                            return Err(bad());
                        }
                    }
                }
                if cursor != sweep.items.len() {
                    return Err(bad());
                }
                let group = sweep.group_update.as_ref().ok_or_else(bad)?;
                if !after_items.valid(ng, map, ns, m1, anchor, cfg)
                    || group.input.as_ref() != Some(&after_items)
                {
                    return Err(bad());
                }
                if group.start != start
                    || group.iteration != iteration
                    || !group.completed
                    || group.algorithm.is_empty()
                    || group.backend.is_empty()
                    || group.device_name.is_empty()
                    || (iteration + 1 == k && converged && group.iteration_limited)
                {
                    return Err(bad());
                }
                let mut expected = Vec::new();
                for g in 1..ng {
                    expected.push((g, None, false));
                    expected.push((g, None, true));
                    for s in 0..ns {
                        if cfg.estimate_specific_means {
                            expected.push((g, Some(s), false));
                        }
                        if cfg.estimate_specific_vars {
                            expected.push((g, Some(s), true));
                        }
                    }
                }
                if expected.len() != group.coordinates.len() {
                    return Err(bad());
                }
                let mut after_group = after_items;
                for (key, r) in expected.iter().zip(&group.coordinates) {
                    let owner = if group.algorithm == "finite_grid_conditional_em" {
                        "gpu"
                    } else if group.algorithm == "posterior_normal_moments"
                        && r.specific.is_some()
                        && !r.is_sd
                        && !cfg.estimate_specific_vars
                    {
                        "cpu_f64"
                    } else if group.algorithm == "posterior_normal_moments" {
                        "gpu"
                    } else {
                        return Err(bad());
                    };
                    if *key != (r.group, r.specific, r.is_sd)
                        || r.owner != owner
                        || !r.value.is_finite()
                        || (r.is_sd && r.value <= 0.0)
                    {
                        return Err(bad());
                    }
                    after_group.set_coordinate(r);
                }
                // The next actual same-start E-step consumes exactly the updated
                // item bank and group coordinates, including fixed coordinates.
                let next = self.dispatch.get(sweep_cursor).ok_or_else(bad)?;
                if next.start != start
                    || next.iteration != iteration + 1
                    || next.input.as_ref() != Some(&after_group)
                {
                    return Err(bad());
                }
            }
        }
        if sweep_cursor != self.dispatch.len()
            || helper_count != self.m_step_gpu_dispatches
            || e.final_posterior.len() != 1
        {
            return Err(bad());
        }
        let f = &e.final_posterior[0];
        let winning = winning_terminal.ok_or_else(bad)?;
        if f.input.as_ref() != Some(winning) || !winning.matches_returned(self, map, ns, anchor) {
            return Err(bad());
        }
        if f.start != self.best_start
            || f.iteration != self.n_iter
            || f.n_persons != gid.len()
            || f.n_items != ni
            || f.n_specific != ns
            || f.q_general != cfg.q_general
            || f.q_specific != cfg.q_specific
            || !f.gpu_used
            || !f.output_consumed
            || f.backend.as_deref().is_none_or(str::is_empty)
            || f.device_name.as_deref().is_none_or(str::is_empty)
            || f.fallback_reason.is_some()
        {
            return Err(bad());
        }
        Ok(())
    }
}
