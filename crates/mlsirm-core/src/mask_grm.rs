//! Concurrent mask GRM inference in the library.
#[path = "mask_grm_likelihood.rs"]
mod likelihood_only;
#[path = "mask_grm_lifecycle.rs"]
pub mod lifecycle;
#[path = "mask_grm_state.rs"]
pub mod state;
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BifactorSpecification {
    specific_map: Vec<i32>,
    n_specific: usize,
    q_specific: usize,
    estimate_specific_means: bool,
    estimate_specific_vars: bool,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    y: Vec<usize>,
    observed: Option<Vec<bool>>,
    group_id: Vec<usize>,
    n_groups: usize,
    n_cat: usize,
    item_ids: Vec<String>,
    anchor: Vec<bool>,
    q_general: usize,
    max_iter: usize,
    tol: f64,
    n_starts: usize,
    seed: u64,
    newton_iter: usize,
    ridge: f64,
    device: String,
    controls: InferenceControls,
    studied_item: Option<usize>,
    alpha: f64,
    max_rounds: usize,
    bifactor: Option<BifactorSpecification>,
}
pub fn run_json(request: &str) -> Result<String, String> {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    let r: Request = serde_json::from_str(request).map_err(|_| "invalid request JSON".to_string())?;
    let n = r.item_ids.len();
    let unique: std::collections::HashSet<_> = r.item_ids.iter().collect();
    if n == 0
        || unique.len() != n
        || r.item_ids.iter().any(|s| s.is_empty())
        || r.anchor.len() != n
        || r.n_groups < 2
        || r.studied_item.is_some_and(|j| j >= n)
    {
        return Err("invalid item/group/constraint identity".into());
    }
    let device = match r.device.as_str() {
        "cpu" => crate::Device::Cpu,
        "gpu" => crate::Device::Gpu,
        _ => return Err("device must be explicit cpu or gpu".into()),
    };
    if r.controls.require_gpu && device != crate::Device::Gpu {
        return Err("GPU inference requires explicit GPU fit".into());
    }
    // Admission of numeric controls precedes any fitter, including single-pair mode.
    if !r.controls.fd_step.is_finite()
        || r.controls.fd_step <= 0.0
        || !r.controls.rank_atol.is_finite()
        || r.controls.rank_atol < 0.0
        || !r.controls.rank_rtol.is_finite()
        || r.controls.rank_rtol < 0.0
        || r.controls.rank_atol + r.controls.rank_rtol <= 0.0
        || r.controls.objective_call_budget == 0
        || r.controls.workspace_bytes == 0
        || !r.alpha.is_finite()
        || r.alpha <= 0.0
        || r.alpha >= 1.0
        || r.max_rounds == 0
    {
        return Err("invalid inference/purification controls".into());
    }
    for v in [
        r.controls.ll_roundoff_tolerance,
        r.controls.fit_ll_tolerance,
        r.controls.lr_backend_tolerance,
        r.controls.hessian_rtol,
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err("invalid likelihood/information tolerance".into());
        }
    }
    if let Some(b) = &r.bifactor {
        if b.n_specific == 0 || b.specific_map.len() != n || b.q_specific == 0 {
            return Err("invalid bifactor specification".into());
        }
    }
    let cfg = BifactorMultigroupConfig {
        q_general: r.q_general,
        q_specific: r.bifactor.as_ref().map_or(1, |b| b.q_specific),
        max_iter: r.max_iter,
        tol: r.tol,
        n_starts: r.n_starts,
        seed: r.seed,
        newton_iter: r.newton_iter,
        ridge: r.ridge,
        estimate_specific_vars: r.bifactor.as_ref().is_some_and(|b| b.estimate_specific_vars),
        estimate_specific_means: r.bifactor.as_ref().is_some_and(|b| b.estimate_specific_means),
        device,
        slope_prior: crate::bifactor_grm::SlopePrior::None,
    };
    let specific_map = r
        .bifactor
        .as_ref()
        .map_or_else(|| vec![-1; n], |b| b.specific_map.clone());
    let n_specific = r.bifactor.as_ref().map_or(0, |b| b.n_specific);
    let result = if let Some(j) = r.studied_item {
        let mut h0 = r.anchor.clone();
        h0[j] = true;
        let null = MaskFit::fit_bifactor(
            &r.y,
            r.observed.as_deref(),
            &r.group_id,
            r.n_groups,
            n,
            r.n_cat,
            &h0,
            &cfg,
            &specific_map,
            n_specific,
        )?;
        match null.release_item(j) {
            Ok(alt) => null.nested_lr(&alt, j, &r.controls)?,
            Err(e) => {
                let summary = null.summary()?;
                json!({"status":"alternative_failed","error":e,"null":summary,"p_value":null})
            }
        }
    } else {
        serde_json::to_value(purify_bifactor_full_matrix(
            &r.y,
            r.observed.as_deref(),
            &r.group_id,
            r.n_groups,
            n,
            r.n_cat,
            &r.anchor,
            &cfg,
            &r.controls,
            r.alpha,
            r.max_rounds,
            &specific_map,
            n_specific,
        )?)
        .map_err(|e| e.to_string())?
    };
    let output = json!({"schema_version":1,"request_sha256":format!("{:x}",Sha256::digest(request.as_bytes())),"core_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("mask_grm.rs"))),"fitter_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("bifactor_grm.rs"))),"model":if r.bifactor.is_some() {"concurrent_bifactor_GRM"} else {"concurrent_general_only_GRM"},"reference_group":0,"n_groups":r.n_groups,"item_ids":r.item_ids,"n_persons":r.group_id.len(),"result":result});
    serde_json::to_string(&output).map_err(|e| e.to_string())
}



use crate::bifactor_grm::{
    fit_bifactor_grm_multigroup, BifactorMultigroupConfig, BifactorMultigroupResult,
};
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskFit {
    y: Vec<usize>,
    observed: Option<Vec<bool>>,
    groups: Vec<usize>,
    n_groups: usize,
    n_items: usize,
    n_cat: usize,
    anchor: Vec<bool>,
    cfg: BifactorMultigroupConfig,
    result: BifactorMultigroupResult,
    specific_map: Vec<i32>,
    n_specific: usize,
    embedded_start: Option<f64>,
    embedded_parent: Option<String>,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Information {
    pub matrix: Vec<f64>,
    pub eigenvalues: Vec<f64>,
    pub dimension: usize,
    pub rank: usize,
    pub negative_eigenvalues: usize,
    pub min_eigenvalue: f64,
    pub threshold: f64,
    pub rank_atol: f64,
    pub rank_rtol: f64,
    pub step: f64,
    pub objective_calls: usize,
    pub fit_identity: String,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceControls {
    pub fd_step: f64,
    pub rank_atol: f64,
    pub rank_rtol: f64,
    pub objective_call_budget: usize,
    pub workspace_bytes: usize,
    pub ll_roundoff_tolerance: f64,
    pub fit_ll_tolerance: f64,
    pub lr_backend_tolerance: f64,
    pub hessian_rtol: f64,
    pub require_gpu: bool,
}
#[derive(Debug, serde::Serialize)]
pub struct PurificationRound {
    pub anchors_before: Vec<bool>,
    pub item_results: Vec<serde_json::Value>,
    pub adjusted_p: Option<Vec<f64>>,
    pub removed: Option<usize>,
    pub status: String,
}
#[derive(Debug, serde::Serialize)]
pub struct PurificationResult {
    pub rounds: Vec<PurificationRound>,
    pub final_anchor: Vec<bool>,
    pub status: String,
}
// Optional aggregate-only execution journal. The caller owns durable capture.
// No respondent rows/scores are emitted. Disabled by default.
pub(crate) fn progress_event(value: serde_json::Value) -> Result<(), String> {
    if std::env::var("FAST_MLSIRM_DIF_PROGRESS").as_deref() == Ok("1") {
        use std::io::Write;
        let mut stderr = std::io::stderr().lock();
        writeln!(stderr, "FMLSIRM_DIF_PROGRESS {}", value)
            .and_then(|_| stderr.flush())
            .map_err(|e| format!("DIF progress capture failed: {e}"))?;
    }
    Ok(())
}
pub fn purify_full_matrix(
    y: &[usize],
    observed: Option<&[bool]>,
    groups: &[usize],
    n_groups: usize,
    n_items: usize,
    n_cat: usize,
    initial_anchor: &[bool],
    cfg: &BifactorMultigroupConfig,
    controls: &InferenceControls,
    alpha: f64,
    max_rounds: usize,
) -> Result<PurificationResult, String> {
    purify_bifactor_full_matrix(
        y,
        observed,
        groups,
        n_groups,
        n_items,
        n_cat,
        initial_anchor,
        cfg,
        controls,
        alpha,
        max_rounds,
        &vec![-1; n_items],
        0,
    )
}

pub fn purify_bifactor_full_matrix(
    y: &[usize],
    observed: Option<&[bool]>,
    groups: &[usize],
    n_groups: usize,
    n_items: usize,
    n_cat: usize,
    initial_anchor: &[bool],
    cfg: &BifactorMultigroupConfig,
    controls: &InferenceControls,
    alpha: f64,
    max_rounds: usize,
    specific_map: &[i32],
    n_specific: usize,
) -> Result<PurificationResult, String> {
    use serde_json::json;
    if initial_anchor.len() != n_items
        || specific_map.len() != n_items
        || n_items == 0
        || !initial_anchor.iter().any(|&a| a)
        || max_rounds == 0
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha >= 1.0
    {
        return Err("invalid purification controls".into());
    }
    let mut anchors = initial_anchor.to_vec();
    let mut rounds = vec![];
    for _ in 0..max_rounds {
        let before = anchors.clone();
        progress_event(
            json!({"event":"round_start","round":rounds.len()+1,"family_size":n_items,"anchors":anchors}),
        )?;
        let mut tests = vec![];
        // The cache is local to this round. Responses and prior nonanchors are
        // never subsetted; an excluded item is tested again on every round.
        let mut cache: std::collections::HashMap<Vec<bool>, MaskFit> =
            std::collections::HashMap::new();
        for j in 0..n_items {
            if !anchors.iter().enumerate().any(|(i, &a)| i != j && a) {
                tests.push(json!({"tested_item":j,"status":"identification_exhausted","error":"H1 has no common linking item","p_value":null}));
                progress_event(
                    json!({"event":"pair_complete","round":rounds.len()+1,"family_size":n_items,"tested_item":j,"result":tests.last()}),
                )?;
                continue;
            }
            let mut h0 = anchors.clone();
            h0[j] = true;
            let result = (|| -> Result<serde_json::Value, String> {
                if !cache.contains_key(&h0) {
                    cache.insert(
                        h0.clone(),
                        MaskFit::fit_bifactor(y, observed, groups, n_groups, n_items, n_cat, &h0, cfg, specific_map, n_specific)?,
                    );
                }
                let null = &cache[&h0];
                let alt = null.release_item(j)?;
                null.nested_lr(&alt, j, controls)
            })();
            match result {
                Ok(r) => tests.push(r),
                Err(e) => {
                    // An explicit GPU failure never starts a later CPU E-step.
                    // Error outcomes remain in the family and prevent adjustment.
                    tests.push(
                        json!({"tested_item":j,"status":"pair_failed","error":e,"p_value":null}),
                    );
                }
            }
            progress_event(
                json!({"event":"pair_complete","round":rounds.len()+1,"family_size":n_items,"tested_item":j,"result":tests.last()}),
            )?;
        }
        let raw: Option<Vec<f64>> = tests
            .iter()
            .map(|r| {
                if r["status"] == "complete" {
                    r["p_value"].as_f64()
                } else {
                    None
                }
            })
            .collect();
        let (adjusted, removed, status) = if let Some(raw) = raw {
            let (adj, removed) = select_removal(&anchors, &raw, alpha)?;
            let status = if removed.is_some() {
                "retest_all"
            } else {
                "stable_no_further_removal"
            };
            (Some(adj), removed, status)
        } else {
            let status = if tests
                .iter()
                .any(|r| r["status"] == "identification_exhausted")
            {
                "identification_exhausted"
            } else {
                "round_incomplete"
            };
            (None, None, status)
        };
        if let Some(j) = removed {
            anchors[j] = false;
        }
        rounds.push(PurificationRound {
            anchors_before: before,
            item_results: tests,
            adjusted_p: adjusted,
            removed,
            status: status.into(),
        });
        progress_event(
            json!({"event":"round_end","round":rounds.len(),"family_size":n_items,"result":rounds.last()}),
        )?;
        if status != "retest_all" {
            return Ok(PurificationResult {
                rounds,
                final_anchor: anchors,
                status: status.into(),
            });
        }
        if !anchors.iter().any(|&a| a) {
            return Ok(PurificationResult {
                rounds,
                final_anchor: anchors,
                status: "identification_exhausted".into(),
            });
        }
    }
    Ok(PurificationResult {
        rounds,
        final_anchor: anchors,
        status: "round_limit_incomplete".into(),
    })
}
/// Pure policy utility, NOT a replacement for admitted nested model tests.
pub fn select_removal(
    anchor: &[bool],
    raw: &[f64],
    alpha: f64,
) -> Result<(Vec<f64>, Option<usize>), String> {
    if anchor.len() != raw.len()
        || anchor.is_empty()
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha >= 1.0
    {
        return Err("invalid complete family".into());
    }
    let adjusted = crate::multiplicity::holm_adjusted_p(raw)?;
    let mut candidates: Vec<usize> = (0..anchor.len())
        .filter(|&i| anchor[i] && adjusted[i] < alpha)
        .collect();
    candidates.sort_by(|&a, &b| {
        adjusted[a]
            .total_cmp(&adjusted[b])
            .then(raw[a].total_cmp(&raw[b]))
            .then(a.cmp(&b))
    });
    Ok((adjusted, candidates.first().copied()))
}
    /// Necessary focal-location link, in squared-loading (Gram eigenvalue)
    /// units. A full rank here is NOT full parameter identification.
    pub(crate) fn anchor_location_link_from_loadings(anchor: &[bool], a_general: &[f64], a_specific: &[f64], specific_map: &[i32], n_specific: usize, estimate_specific_means: bool, atol: f64, rtol: f64) -> Result<serde_json::Value, String> {
        if !atol.is_finite() || !rtol.is_finite() || atol < 0.0 || rtol < 0.0
            || (atol == 0.0 && rtol == 0.0) {
            return Err("invalid anchor location rank controls".into());
        }
        if anchor.is_empty() || anchor.len() != a_general.len() || anchor.len() != a_specific.len()
            || anchor.len() != specific_map.len() || specific_map.iter().any(|&s| s < -1 || (s >= 0 && s as usize >= n_specific)) {
            return Err("invalid anchor location geometry".into());
        }
        let dim = 1 + if estimate_specific_means {n_specific} else {0};
        let mut gram = vec![0.0; dim * dim];
        let mut rows = 0;
        for (i, &anchor) in anchor.iter().enumerate() {
            if !anchor {continue;}
            rows += 1;
            let mut row = vec![0.0;dim];
            row[0] = a_general[i];
            if dim > 1 && specific_map[i] >= 0 {
                row[1 + specific_map[i] as usize] = a_specific[i];
            }
            if row.iter().any(|v| !v.is_finite()) {return Err("nonfinite anchor loading".into());}
            for j in 0..dim {for k in 0..dim {gram[j*dim+k] += row[j]*row[k];}}
        }
        let (_, _, eigenvalues) = crate::inference::second_order_test(&gram,dim,0.0)?;
        let scale = eigenvalues.iter().map(|v|v.abs()).fold(0.0,f64::max);
        let threshold = atol + rtol * scale;
        if !threshold.is_finite() {return Err("anchor rank threshold overflow".into());}
        let rank = eigenvalues.iter().filter(|&&v|v > threshold).count();
        Ok(serde_json::json!({"rank":rank,"dimension":dim,"anchor_count":rows,
            "gram_eigenvalues":eigenvalues,"threshold":threshold,"rank_atol":atol,"rank_rtol":rtol,
            "threshold_units":"squared_loading","scope":"necessary_focal_location_link_only"}))
    }

impl MaskFit {
    pub fn anchor_location_link(&self, atol: f64, rtol: f64) -> Result<serde_json::Value, String> {
        anchor_location_link_from_loadings(&self.anchor,&self.result.a_general[0],&self.result.a_specific[0],
            &self.specific_map,self.n_specific,self.cfg.estimate_specific_means,atol,rtol)
    }

    /// The only LR producer consumes private library-owned fits, recomputes
    /// information itself, and never accepts caller-supplied rank/pass flags.
    pub fn nested_lr(
        &self,
        alt: &Self,
        item: usize,
        c: &InferenceControls,
    ) -> Result<serde_json::Value, String> {
        use serde_json::json;
        for v in [
            c.ll_roundoff_tolerance,
            c.fit_ll_tolerance,
            c.lr_backend_tolerance,
            c.hessian_rtol,
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err("invalid LR numerical tolerance".into());
            }
        }
        if self.y != alt.y
            || self.observed != alt.observed
            || self.groups != alt.groups
            || self.n_groups != alt.n_groups
            || self.n_items != alt.n_items
            || self.n_cat != alt.n_cat
            || self.specific_map != alt.specific_map
            || self.n_specific != alt.n_specific
            || format!("{:?}", self.cfg) != format!("{:?}", alt.cfg)
        {
            return Err("nested model input/settings identity mismatch".into());
        }
        if item >= self.n_items
            || !self.anchor[item]
            || alt.anchor[item]
            || self
                .anchor
                .iter()
                .zip(&alt.anchor)
                .enumerate()
                .any(|(i, (a, b))| i != item && a != b)
        {
            return Err("nested constraint difference must be exactly the tested item".into());
        }
        if alt.embedded_parent.as_deref() != Some(self.identity()?.as_str()) {
            return Err("alternative is not bound to this embedded null".into());
        }
        let x0 = self.packed()?;
        let x1 = alt.packed()?;
        let p0 = x0.len();
        let p1 = x1.len();
        if p0 != self.result.n_parameters || p1 != alt.result.n_parameters {
            return Err("fitter/free-coordinate count mismatch".into());
        }
        let map0 = self.coordinate_map()?;
        let map1 = alt.coordinate_map()?;
        let mut embedded = Vec::with_capacity(p1);
        for coordinate in &map1 {
            let source = match *coordinate {
                Coordinate::Slope(_, i) if i == item => Coordinate::Slope(0, i),
                Coordinate::SpecificSlope(_, i) if i == item => Coordinate::SpecificSlope(0, i),
                Coordinate::FirstIntercept(_, i) if i == item => Coordinate::FirstIntercept(0, i),
                Coordinate::LogGap(_, i, k) if i == item => Coordinate::LogGap(0, i, k),
                _ => coordinate.clone(),
            };
            embedded.push(
                x0[map0
                    .iter()
                    .position(|v| v == &source)
                    .ok_or("embedding coordinate absent")?],
            );
        }
        let ll0 = self.loglik_at(&x0)?;
        let ll1 = alt.loglik_at(&x1)?;
        let embedded_ll = alt.loglik_at(&embedded)?;
        let raw = 2.0 * (ll1 - ll0);
        let device_lr = 2.0 * (alt.trace_loglik() - self.trace_loglik());
        let mut output = json!({"status":"unverified","p_value":null,"df":null,"tested_item":item,
            "null":self.summary()?,"alternative":alt.summary()?,"raw_lr_cpu_f64":raw,
            "raw_lr_fit_backend":device_lr,"embedding_loglik":embedded_ll,
            "information_device":"cpu_f64", "inference_scope":"asymptotic_conditional_on_current_anchors",
            "controls":c,"information":[],"effects":"not_computed"});
        let mut refusal = None;
        if (embedded_ll - ll0).abs() > c.ll_roundoff_tolerance
            || alt
                .embedded_start
                .is_none_or(|v| (v - self.trace_loglik()).abs() > c.fit_ll_tolerance)
        {
            refusal = Some("embedding_likelihood_mismatch");
        } else if !self.result.converged
            || !alt.result.converged
            || self.result.termination_reason != "tolerance_met"
            || alt.result.termination_reason != "tolerance_met"
        {
            refusal = Some("nonconverged");
        } else if c.require_gpu
            && ![self, alt].iter().all(|f| {
                f.cfg.device == crate::Device::Gpu
                    && f.result.dispatch_complete
                    && !f.result.dispatch.is_empty()
                    && f.result.dispatch.iter().all(|d| {
                        d.gpu_used
                            && d.backend.is_some()
                            && d.device_name.is_some()
                            && d.fallback_reason.is_none()
                    })
            })
        {
            refusal = Some("gpu_receipt_required");
        } else if (ll0 - self.trace_loglik()).abs() > c.fit_ll_tolerance
            || (ll1 - alt.trace_loglik()).abs() > c.fit_ll_tolerance
            || (raw - device_lr).abs() > c.lr_backend_tolerance
        {
            refusal = Some("backend_likelihood_sensitivity");
        } else if raw < -c.ll_roundoff_tolerance || device_lr < -c.ll_roundoff_tolerance {
            refusal = Some("negative_likelihood_difference");
        } else if raw < 0.0 || device_lr < 0.0 {
            refusal = Some("negative_roundoff_requires_review");
        }
        if let Some(reason) = refusal {
            output["status"] = json!(reason);
            return Ok(output);
        }
        if self.n_specific == 0 {
            // Preserve the general-only scalar link contract. Its rank_rtol
            // controls information eigenvalues, not this absolute slope check.
            if !self.anchor.iter().enumerate().any(|(i, &a)|
                a && i != item && self.result.a_general[0][i].abs() > c.rank_atol) {
                output["status"] = json!("linking_rank_deficient");
                return Ok(output);
            }
        } else {
            // Both bifactor loading matrices must link every free focal mean.
            // This location-only necessary condition never replaces information.
            let links = [self.anchor_location_link(c.rank_atol, c.rank_rtol)?,
                alt.anchor_location_link(c.rank_atol, c.rank_rtol)?];
            output["anchor_location_link"] = json!(links);
            if links.iter().any(|v| v["rank"] != v["dimension"]) {
                output["status"] = json!("linking_rank_deficient");
                return Ok(output);
            }
        }
        let mut infos = Vec::new();
        for f in [self, alt] {
            for step in [c.fd_step, c.fd_step * 0.5] {
                match f.information(
                    step,
                    c.rank_atol,
                    c.rank_rtol,
                    c.objective_call_budget,
                    c.workspace_bytes,
                ) {
                    Ok(info) => infos.push(info),
                    Err(e) => {
                        output["status"] = json!("information_unavailable");
                        output["information_error"] = json!(e);
                        output["information"] = json!(infos);
                        return Ok(output);
                    }
                }
            }
        }
        let valid = infos.iter().all(|i| {
            i.rank == i.dimension && i.negative_eigenvalues == 0 && i.min_eigenvalue > i.threshold
        });
        let stable = [(0, 1), (2, 3)].iter().all(|&(a, b)| {
            let scale = infos[a].matrix.iter().map(|v| v.abs()).fold(0.0, f64::max);
            infos[a].rank == infos[b].rank
                && infos[a]
                    .matrix
                    .iter()
                    .zip(&infos[b].matrix)
                    .all(|(x, y)| (x - y).abs() <= c.rank_atol + c.hessian_rtol * scale)
        });
        output["information"] = json!(infos);
        if !valid {
            output["status"] = json!("information_not_positive_full_rank");
            return Ok(output);
        }
        if !stable {
            output["status"] = json!("information_step_unstable");
            return Ok(output);
        }
        let df = p1
            .checked_sub(p0)
            .ok_or("invalid nested parameter dimension")?;
        if df == 0 || df != (self.n_groups - 1) * (self.n_cat + usize::from(self.specific_map[item] >= 0)) {
            return Err("released coordinate df mismatch".into());
        }
        output["df"] = json!(df);
        output["p_value"] = json!(crate::fitstats::chi2_sf(raw, df as f64));
        output["status"] = json!("complete");
        Ok(output)
    }
    fn identity(&self) -> Result<String, String> {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(&(
            &self.y,
            &self.observed,
            &self.groups,
            self.n_groups,
            self.n_items,
            self.n_cat,
            &self.anchor,
            &self.specific_map, self.n_specific,
            format!("{:?}", self.cfg),
            self.packed()?,
        ))
        .map_err(|e| e.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
    pub fn summary(&self) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let r = &self.result;
        let dispatch:Vec<_>=r.dispatch.iter().map(|d|json!({"start":d.start,"iteration":d.iteration,"gpu_used":d.gpu_used,"backend":d.backend,"device_name":d.device_name,"fallback_reason":d.fallback_reason})).collect();
        Ok(
            json!({"fit_identity":self.identity()?,"anchor":self.anchor,"coordinate_map":self.coordinate_map()?,"packed":self.packed()?,"loglik_cpu_f64":self.loglik_at(&self.packed()?)?,"loglik_trace":r.loglik_trace,"converged":r.converged,"termination_reason":r.termination_reason,"n_iter":r.n_iter,"final_loglik_change":r.final_loglik_change,"stopping_rule":"delta <= tol*(1+abs(previous_LL))","tol":self.cfg.tol,"n_parameters":r.n_parameters,"best_start":r.best_start,"q_general":self.cfg.q_general,"general_mean":r.general_mean,"general_sd":r.general_sd,"specific_mean":r.specific_mean,"specific_sd":r.specific_sd,"specific_map":self.specific_map,"n_specific":self.n_specific,"q_specific":self.cfg.q_specific,"estimate_specific_means":self.cfg.estimate_specific_means,"estimate_specific_vars":self.cfg.estimate_specific_vars,"dispatch":dispatch,"dispatch_complete":r.dispatch_complete,"starts":r.start_outcomes,"start_selection":"maximum_terminal_LL; reject nonconverged winner","m_step_device":if self.cfg.device == crate::Device::Gpu {"gpu"} else {"cpu_f64"},"m_step_gpu_dispatches":r.m_step_gpu_dispatches,"final_eap_device":"cpu_f64","final_posterior_device":if r.scoring_posterior_gpu {"gpu_two_word"} else {"cpu_f64"},"embedded_start":self.embedded_start,"embedded_parent":self.embedded_parent}),
        )
    }

    pub fn information(
        &self,
        step: f64,
        atol: f64,
        rtol: f64,
        call_budget: usize,
        workspace_bytes: usize,
    ) -> Result<Information, String> {
        self.information_impl(step, atol, rtol, call_budget, workspace_bytes, true)
    }
    fn information_impl(&self, step: f64, atol: f64, rtol: f64, call_budget: usize, workspace_bytes: usize, likelihood_only: bool) -> Result<Information, String> {
        if !step.is_finite()
            || step <= 0.0
            || !atol.is_finite()
            || atol < 0.0
            || !rtol.is_finite()
            || rtol < 0.0
            || (atol == 0.0 && rtol == 0.0)
        {
            return Err("invalid information controls".into());
        }
        let x = self.packed()?;
        let n = x.len();
        let square = n.checked_mul(n).ok_or("information dimension overflow")?;
        let calls = square
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or("call budget overflow")?;
        // Conservative peak allowance: stencil, matrix, eigen workspaces, model probability tables.
        let tables = self
            .n_groups
            .checked_mul(self.n_items)
            .and_then(|v| v.checked_mul(self.n_cat))
            .and_then(|v| v.checked_mul(self.cfg.q_general))
            .and_then(|v| v.checked_mul(if self.n_specific > 0 {self.cfg.q_specific} else {1}))
            .ok_or("table overflow")?;
        // Bounded likelihood cache: base + last per factor, plus one live
        // replacement. Include row/key/index metadata conservatively in f64 units.
        let cache_units = if self.n_specific > 0 && likelihood_only {
            self.groups.len().checked_mul(self.cfg.q_general)
                .and_then(|v|v.checked_mul(self.n_specific.checked_add(1)?))
                .and_then(|v|v.checked_mul(3))
                .and_then(|v|v.checked_add(self.groups.len().checked_mul(8)?))
                .and_then(|v|v.checked_add(self.n_groups.checked_mul(self.n_specific.checked_add(1)?)?.checked_mul(32)?))
                .and_then(|v|v.checked_add(self.n_groups.checked_mul(self.n_items)?.checked_mul(self.n_cat.checked_add(4)?)?.checked_mul(4)?))
                .ok_or("likelihood cache overflow")?
        } else {0};
        let bytes = square
            .checked_mul(16)
            .and_then(|v| v.checked_add(tables))
            .and_then(|v| v.checked_add(cache_units))
            .and_then(|v| v.checked_add(16 * n))
            .and_then(|v| v.checked_mul(8))
            .ok_or("workspace overflow")?;
        if calls > call_budget || bytes > workspace_bytes {
            return Err(format!(
                "information budget exceeded: calls={calls}, workspace={bytes}"
            ));
        }
        let mut cache = if self.n_specific > 0 && likelihood_only {
            Some(self::likelihood_only::Cache::new(self)?)
        } else {None};
        let mut objective = |z: &[f64]| self.loglik_at_cached(z, likelihood_only, cache.as_mut()).map(|v| -v);
        let base = objective(&x)?;
        let mut plus = vec![];
        let mut minus = vec![];
        for i in 0..n {
            let mut z = x.clone();
            z[i] += step;
            plus.push(objective(&z)?);
            z[i] = x[i] - step;
            minus.push(objective(&z)?);
        }
        let mut off = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for i in 0..n {
            for j in i + 1..n {
                for (v, (si, sj)) in
                    off.iter_mut()
                        .zip([(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)])
                {
                    let mut z = x.clone();
                    z[i] += si * step;
                    z[j] += sj * step;
                    v.push(objective(&z)?);
                }
            }
        }
        let matrix = crate::inference::finite_difference_hessian(
            n, step, base, &plus, &minus, &off[0], &off[1], &off[2], &off[3],
        )?;
        let (_, min_eigenvalue, eigenvalues) =
            crate::inference::second_order_test(&matrix, n, 0.0)?;
        let scale = eigenvalues.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let threshold = atol + rtol * scale;
        let rank = eigenvalues.iter().filter(|v| v.abs() > threshold).count();
        let negative_eigenvalues = eigenvalues.iter().filter(|&&v| v < -threshold).count();
        let fit_identity = self.identity()?;
        Ok(Information {
            matrix,
            eigenvalues,
            dimension: n,
            rank,
            negative_eigenvalues,
            min_eigenvalue,
            threshold,
            rank_atol: atol,
            rank_rtol: rtol,
            step,
            objective_calls: calls,
            fit_identity,
        })
    }

    pub fn fit(
        y: &[usize],
        observed: Option<&[bool]>,
        groups: &[usize],
        n_groups: usize,
        n_items: usize,
        n_cat: usize,
        anchor: &[bool],
        cfg: &BifactorMultigroupConfig,
    ) -> Result<Self, String> {
        Self::fit_bifactor(y, observed, groups, n_groups, n_items, n_cat, anchor, cfg, &vec![-1;n_items], 0)
    }
    /// Fit a preserved full bifactor item map; no response column subsetting.
    pub fn fit_bifactor(
        y: &[usize], observed: Option<&[bool]>, groups: &[usize], n_groups: usize,
        n_items: usize, n_cat: usize, anchor: &[bool], cfg: &BifactorMultigroupConfig,
        specific_map: &[i32], n_specific: usize,
    ) -> Result<Self, String> {
        if n_groups < 2 {return Err("mask inference requires multiple groups".into());}
        progress_event(serde_json::json!({"event":"fit_start","anchor":anchor,"n_starts":cfg.n_starts,"seed":cfg.seed,"n_persons":groups.len(),"n_items":n_items,"n_specific":n_specific,"q_general":cfg.q_general,"q_specific":cfg.q_specific}))?;
        let result = fit_bifactor_grm_multigroup(
            y,
            observed,
            groups,
            n_groups,
            specific_map,
            groups.len(),
            n_items,
            n_specific,
            n_cat,
            Some(anchor),
            cfg,
        )?;
        Ok(Self {
            y: y.to_vec(),
            observed: observed.map(|x| x.to_vec()),
            groups: groups.to_vec(),
            n_groups,
            n_items,
            n_cat,
            anchor: anchor.to_vec(),
            cfg: *cfg,
            result,
            specific_map: specific_map.to_vec(),
            n_specific,
            embedded_start: None,
            embedded_parent: None,
        })
    }
    pub fn coordinate_map(&self) -> Result<Vec<Coordinate>, String> {
        bifactor_coordinates(&self.anchor,self.n_groups,self.n_cat,&self.specific_map,self.n_specific,
            self.cfg.estimate_specific_means,self.cfg.estimate_specific_vars)
    }
    pub fn packed(&self) -> Result<Vec<f64>, String> {
        let r = &self.result;
        let m = self.n_cat - 1;
        let x: Vec<f64> = self.coordinate_map()?
            .iter()
            .map(|c| match *c {
                Coordinate::Slope(g, i) => r.a_general[g][i],
                Coordinate::SpecificSlope(g, i) => r.a_specific[g][i],
                Coordinate::SpecificMean(g, s) => r.specific_mean[g][s],
                Coordinate::SpecificLogSd(g, s) => r.specific_sd[g][s].ln(),
                Coordinate::FirstIntercept(g, i) => r.threshold[g][i * m],
                Coordinate::LogGap(g, i, k) => {
                    (r.threshold[g][i * m + k - 1] - r.threshold[g][i * m + k]).ln()
                }
                Coordinate::Mean(g) => r.general_mean[g],
                Coordinate::LogSd(g) => r.general_sd[g].ln(),
            })
            .collect();
        if x.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite or boundary fit".into());
        }
        Ok(x)
    }
    pub fn loglik_at(&self, x: &[f64]) -> Result<f64, String> {
        self.loglik_at_impl(x, true)
    }
    fn loglik_at_impl(&self, x: &[f64], likelihood_only: bool) -> Result<f64, String> {
        self.loglik_at_cached(x, likelihood_only, None)
    }
    fn loglik_at_cached(&self, x: &[f64], likelihood_only: bool, cache: Option<&mut self::likelihood_only::Cache<'_>>) -> Result<f64, String> {
        let map = self.coordinate_map()?;
        if x.len() != map.len() || x.iter().any(|v| !v.is_finite()) {
            return Err("invalid packed coordinates".into());
        }
        let m = self.n_cat - 1;
        let mut a = vec![vec![0.0; self.n_items]; self.n_groups];
        let mut d = vec![vec![0.0; self.n_items * m]; self.n_groups];
        let mut mu = vec![0.0; self.n_groups];
        let mut sd = vec![1.0; self.n_groups];
        let mut a_s = vec![vec![0.0;self.n_items];self.n_groups];
        let mut mu_s = vec![vec![0.0;self.n_specific];self.n_groups];
        let mut sd_s = vec![vec![1.0;self.n_specific];self.n_groups];
        for (c, &v) in map.iter().zip(x) {
            match *c {
                Coordinate::Slope(g, i) => a[g][i] = v,
                Coordinate::SpecificSlope(g, i) => a_s[g][i] = v,
                Coordinate::SpecificMean(g, s) => mu_s[g][s] = v,
                Coordinate::SpecificLogSd(g, s) => sd_s[g][s] = v.exp(),
                Coordinate::FirstIntercept(g, i) => d[g][i * m] = v,
                Coordinate::LogGap(g, i, k) => d[g][i * m + k] = d[g][i * m + k - 1] - v.exp(),
                Coordinate::Mean(g) => mu[g] = v,
                Coordinate::LogSd(g) => sd[g] = v.exp(),
            }
        }
        for g in 1..self.n_groups {
            for i in 0..self.n_items {
                if self.anchor[i] {
                    a[g][i] = a[0][i];
                    a_s[g][i] = a_s[0][i];
                    for k in 0..m {
                        d[g][i * m + k] = d[0][i * m + k];
                    }
                }
            }
        }
        if sd.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || sd_s.iter().flatten().any(|v| !v.is_finite() || *v <= 0.0)
            || d.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("coordinate transform overflow".into());
        }
        for row in &d {
            for item in row.chunks_exact(m) {
                if item.windows(2).any(|w| w[0] <= w[1]) {
                    return Err("threshold gap underflow".into());
                }
            }
        }
        if self.n_specific > 0 {
            let r = crate::bifactor_grm::MultigroupParameters {
                a_general:&a,a_specific:&a_s,threshold:&d,general_mean:&mu,
                general_sd:&sd,specific_mean:&mu_s,specific_sd:&sd_s,
            };
            let _ = likelihood_only;
            return match cache {Some(c)=>c.loglik(&r),None=>self::likelihood_only::loglik(self, &r)};
        }
        let (nodes, weights) = crate::bifactor_grm::gh_rule(self.cfg.q_general)?;
        let q = nodes.len();
        let mut tables = vec![vec![0.0; self.n_items * q * self.n_cat]; self.n_groups];
        for g in 0..self.n_groups {
            for i in 0..self.n_items {
                for t in 0..q {
                    let probs = crate::poly::grm_logprobs(
                        a[g][i] * (mu[g] + sd[g] * nodes[t]),
                        &d[g][i * m..(i + 1) * m],
                    );
                    tables[g][(i * q + t) * self.n_cat..(i * q + t + 1) * self.n_cat]
                        .copy_from_slice(&probs);
                }
            }
        }
        let mut ll = 0.0;
        let mut terms = vec![0.0; q];
        for (p, &g) in self.groups.iter().enumerate() {
            for t in 0..q {
                terms[t] = weights[t].ln();
                for i in 0..self.n_items {
                    if self
                        .observed
                        .as_ref()
                        .is_none_or(|o| o[p * self.n_items + i])
                    {
                        terms[t] +=
                            tables[g][(i * q + t) * self.n_cat + self.y[p * self.n_items + i]];
                    }
                }
            }
            let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ll += max + terms.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
        }
        if !ll.is_finite() {
            return Err("nonfinite marginal likelihood".into());
        }
        Ok(ll)
    }
    pub fn release_item(&self, item: usize) -> Result<Self, String> {
        if item >= self.n_items || !self.anchor[item] {
            return Err("tested item is not common under H0".into());
        }
        let mut anchor = self.anchor.clone();
        anchor[item] = false;
        let mut embedded_start = None;
        let result = crate::bifactor_grm::fit_bifactor_grm_multigroup_initialized(
            &self.y,
            self.observed.as_deref(),
            &self.groups,
            self.n_groups,
            &self.specific_map,
            self.groups.len(),
            self.n_items,
            self.n_specific,
            self.n_cat,
            Some(&anchor),
            &self.cfg,
            Some(&self.result),
            &mut embedded_start,
        )?;
        Ok(Self {
            y: self.y.clone(),
            observed: self.observed.clone(),
            groups: self.groups.clone(),
            n_groups: self.n_groups,
            n_items: self.n_items,
            n_cat: self.n_cat,
            anchor,
            cfg: self.cfg,
            result,
            specific_map: self.specific_map.clone(),
            n_specific: self.n_specific,
            embedded_start,
            embedded_parent: Some(self.identity()?),
        })
    }
    pub fn embedded_start_loglik(&self) -> Option<f64> {
        self.embedded_start
    }
    pub fn trace_loglik(&self) -> f64 {
        *self.result.loglik_trace.last().unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum Coordinate {
    Slope(usize, usize),
    SpecificSlope(usize, usize),
    SpecificMean(usize, usize),
    SpecificLogSd(usize, usize),
    FirstIntercept(usize, usize),
    LogGap(usize, usize, usize),
    Mean(usize),
    LogSd(usize),
}
pub fn coordinates(
    anchor: &[bool],
    groups: usize,
    categories: usize,
) -> Result<Vec<Coordinate>, String> {
    bifactor_coordinates(anchor,groups,categories,&vec![-1;anchor.len()],0,false,false)
}
pub fn bifactor_coordinates(anchor: &[bool],groups: usize,categories: usize,
    specific_map: &[i32],n_specific: usize,means: bool,vars: bool) -> Result<Vec<Coordinate>,String> {
    if anchor.is_empty() || groups < 2 || categories < 2 || specific_map.len()!=anchor.len()
        || specific_map.iter().any(|&s|s < -1 || (s>=0 && s as usize>=n_specific)) {
        return Err("invalid coordinate geometry".into());
    }
    let mut result = Vec::new();
    for (item, &common) in anchor.iter().enumerate() {
        for g in 0..if common { 1 } else { groups } {
            result.push(Coordinate::Slope(g, item));
            if specific_map[item]>=0 {result.push(Coordinate::SpecificSlope(g,item));}
            result.push(Coordinate::FirstIntercept(g, item));
            for k in 1..categories - 1 {
                result.push(Coordinate::LogGap(g, item, k));
            }
        }
    }
    for g in 1..groups {
        result.push(Coordinate::Mean(g));
        result.push(Coordinate::LogSd(g));
        for s in 0..n_specific {
            if means {result.push(Coordinate::SpecificMean(g,s));}
            if vars {result.push(Coordinate::SpecificLogSd(g,s));}
        }
    }
    Ok(result)
}
