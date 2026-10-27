//! Trusted private fitted-state persistence, not an optimizer checkpoint or attestation.
//! The payload contains responses and scores: callers must use private storage.
use super::*;
use sha2::{Digest, Sha256};

thread_local! {
    static CHECKPOINT_DIR: std::cell::RefCell<Option<std::path::PathBuf>> = const { std::cell::RefCell::new(None) };
}

/// Opt-in private fitted-state journal. Never resumes an optimizer or authenticates execution.
/// A new directory is required; nested scopes are refused and panic unwinding clears ownership.
pub fn with_private_checkpoints<T>(path: &std::path::Path, run: impl FnOnce() -> Result<T,String>) -> Result<T,String> {
    use std::os::unix::fs::DirBuilderExt;
    if CHECKPOINT_DIR.with(|d|d.borrow().is_some()) { return Err("nested checkpoint scope".into()); }
    std::fs::DirBuilder::new().mode(0o700).create(path).map_err(|e|format!("checkpoint directory: {e}"))?;
    CHECKPOINT_DIR.with(|d|*d.borrow_mut()=Some(path.to_owned()));
    struct Clear;
    impl Drop for Clear { fn drop(&mut self) {CHECKPOINT_DIR.with(|d|*d.borrow_mut()=None);} }
    let _clear=Clear;
    run()
}

impl MaskFit {
    /// Persist the complete native typed object before its caller consumes a completed unit.
    pub fn checkpoint_private(&self) -> Result<(),String> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let dir=CHECKPOINT_DIR.with(|d|d.borrow().clone());
        let Some(dir)=dir else {return Ok(());};
        let raw=self.export_private()?;
        let digest=format!("{:x}",Sha256::digest(raw.as_bytes()));
        let dest=dir.join(format!("{digest}.maskfit.json"));
        if dest.exists() {
            return if std::fs::read(&dest).map_err(|e|e.to_string())?==raw.as_bytes() {Ok(())} else {Err("checkpoint collision".into())};
        }
        let tmp=dir.join(format!(".{digest}.tmp"));
        let mut file=std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp).map_err(|e|e.to_string())?;
        file.write_all(raw.as_bytes()).and_then(|_|file.sync_all()).map_err(|e|e.to_string())?;
        // Hard-link publication refuses replacement of an existing checkpoint.
        std::fs::hard_link(&tmp,&dest).map_err(|e|e.to_string())?;
        std::fs::remove_file(&tmp).map_err(|e|e.to_string())?;
        std::fs::File::open(&dir).and_then(|f|f.sync_all()).map_err(|e|e.to_string())?;
        Ok(())
    }

    pub fn export_private(&self) -> Result<String, String> {
        let state = serde_json::to_string(self).map_err(|_| "state serialization failed")?;
        // String payload retains binary64 round-trip spelling and prevents Value conversion loss.
        Ok(serde_json::json!({"schema":1,"source":state_source(),
            "sha256":format!("{:x}",Sha256::digest(state.as_bytes())),"state":state})
        .to_string())
    }
    pub fn restore_private(raw: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope {
            schema: u32,
            source: String,
            sha256: String,
            state: String,
        }
        let e: Envelope = serde_json::from_str(raw).map_err(|_| "invalid private envelope")?;
        if e.schema != 1
            || e.source != state_source()
            || e.sha256 != format!("{:x}", Sha256::digest(e.state.as_bytes()))
        {
            return Err("private state schema/source/digest mismatch".into());
        }
        let f: Self = serde_json::from_str(&e.state).map_err(|_| "invalid private state")?;
        f.validate_private()?;
        let packed = f.packed()?;
        if packed.len() != f.result.n_parameters
            || packed.iter().any(|v| !v.is_finite())
            || !f.loglik_at(&packed)?.is_finite()
        {
            return Err("invalid private fit coordinates".into());
        }
        Ok(f)
    }

    /// Check stored coordinates before packing discards fixed/reference rows.
    /// These are persistence invariants, not authentication or fit attestation.
    fn validate_private(&self) -> Result<(), String> {
        let g = self.n_groups;
        let n = self.n_items;
        let s = self.n_specific;
        let r = &self.result;
        let persons = self.groups.len();
        let cells = persons.checked_mul(n).ok_or("state geometry overflow")?;
        let m = self.n_cat.checked_sub(1).ok_or("invalid categories")?;
        let thresholds = n.checked_mul(m).ok_or("state geometry overflow")?;
        let count_width = n.checked_mul(self.n_cat).ok_or("state geometry overflow")?;
        let matrix = |v: &[Vec<f64>], width: usize| {
            v.len() == g
                && v.iter()
                    .all(|row| row.len() == width && row.iter().all(|x| x.is_finite()))
        };
        if g < 2
            || n == 0
            || persons == 0
            || self.n_cat < 2
            || self.y.len() != cells
            || self.groups.iter().any(|&v| v >= g)
            || self.observed.as_ref().is_some_and(|v| v.len() != cells)
            || self.anchor.len() != n
            || self.specific_map.len() != n
            || self
                .specific_map
                .iter()
                .any(|&v| v < -1 || (v >= 0 && v as usize >= s))
            || !matrix(&r.a_general, n)
            || !matrix(&r.a_specific, n)
            || !matrix(&r.threshold, thresholds)
            || !matrix(&r.specific_mean, s)
            || !matrix(&r.specific_sd, s)
            || r.general_mean.len() != g
            || r.general_sd.len() != g
            || r.theta_g_eap.len() != persons
            || r.theta_g_sd.len() != persons
            || r.group_category_counts.len() != g
            || r.group_category_counts
                .iter()
                .any(|row| row.len() != count_width)
        {
            return Err("invalid private state geometry".into());
        }
        // Match the native missingness contract: only observed values index a
        // probability table or category-count array. Check before either use.
        let observed = |j: usize| self.observed.as_ref().is_none_or(|o| o[j]);
        if self
            .y
            .iter()
            .enumerate()
            .any(|(j, &y)| observed(j) && y >= self.n_cat)
        {
            return Err("observed response categories must be < n_cat".into());
        }
        if self.cfg.q_general == 0
            || self.cfg.q_specific == 0
            || self.cfg.max_iter == 0
            || self.cfg.n_starts == 0
            || self.cfg.newton_iter == 0
            || !self.cfg.tol.is_finite()
            || self.cfg.tol <= 0.0
            || !self.cfg.ridge.is_finite()
            || self.cfg.ridge <= 0.0
            || self.cfg.slope_prior != r.slope_prior
        {
            return Err("invalid private fit settings".into());
        }
        self.cfg.slope_prior.validate()?;
        if r.general_mean.iter().any(|x| !x.is_finite())
            || r.general_sd
                .iter()
                .chain(r.specific_sd.iter().flatten())
                .any(|x| !x.is_finite() || *x <= 0.0)
            || r.general_mean[0] != 0.0
            || r.general_sd[0] != 1.0
        {
            return Err("invalid private latent moments".into());
        }
        for group in 0..g {
            for block in 0..s {
                if ((group == 0 || !self.cfg.estimate_specific_means)
                    && r.specific_mean[group][block] != 0.0)
                    || ((group == 0 || !self.cfg.estimate_specific_vars)
                        && r.specific_sd[group][block] != 1.0)
                {
                    return Err("inconsistent private fixed specific moments".into());
                }
            }
            for item in 0..n {
                if self.specific_map[item] < 0 && r.a_specific[group][item] != 0.0 {
                    return Err("inconsistent private inactive specific slope".into());
                }
                let row = &r.threshold[group][item * m..(item + 1) * m];
                if row.windows(2).any(|w| w[0] <= w[1]) {
                    return Err("invalid private ordered thresholds".into());
                }
                if group > 0
                    && self.anchor[item]
                    && (r.a_general[group][item] != r.a_general[0][item]
                        || r.a_specific[group][item] != r.a_specific[0][item]
                        || row != &r.threshold[0][item * m..(item + 1) * m])
                {
                    return Err("inconsistent private anchor coordinates".into());
                }
            }
        }
        if r.theta_g_eap.iter().any(|x| !x.is_finite())
            || r.theta_g_sd.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("invalid private posterior scores".into());
        }
        // Counts have an exact response-derived meaning; do not normalize them.
        let mut counts = vec![vec![0usize; count_width]; g];
        for (person, &group) in self.groups.iter().enumerate() {
            for item in 0..n {
                let j = person * n + item;
                if observed(j) {
                    counts[group][item * self.n_cat + self.y[j]] += 1;
                }
            }
        }
        if counts != r.group_category_counts {
            return Err("inconsistent private category counts".into());
        }
        if r.loglik_trace.is_empty()
            || r.n_iter.checked_add(1) != Some(r.loglik_trace.len())
            || r.em_objective_trace.len() != r.loglik_trace.len()
            || r.n_iter > self.cfg.max_iter
            || r.loglik_trace
                .iter()
                .chain(&r.em_objective_trace)
                .any(|x| !x.is_finite())
            || !r.final_loglik_change.is_finite()
        {
            return Err("invalid private fit trace".into());
        }
        Ok(())
    }
}
fn state_source() -> String {
    let mut h = Sha256::new();
    h.update(include_bytes!("mask_grm.rs"));
    h.update(include_bytes!("bifactor_grm.rs"));
    h.update(include_bytes!("mask_grm_state.rs"));
    format!("{:x}", h.finalize())
}

/// Explicit private-state output; no printing and no automatic filesystem writes.
pub fn fit_private(request: &str) -> Result<String, String> {
    let r: Request = serde_json::from_str(request).map_err(|_| "invalid request JSON")?;
    let n = r.item_ids.len();
    if n == 0 || r.anchor.len() != n {
        return Err("invalid item identity".into());
    }
    let device = match r.device.as_str() {
        "cpu" => crate::Device::Cpu,
        "gpu" => crate::Device::Gpu,
        _ => return Err("explicit device required".into()),
    };
    if r.controls.require_gpu && device != crate::Device::Gpu {
        return Err("GPU required".into());
    }
    let (map, ns, qs, means, vars) = match &r.bifactor {
        Some(b) => (
            b.specific_map.clone(),
            b.n_specific,
            b.q_specific,
            b.estimate_specific_means,
            b.estimate_specific_vars,
        ),
        None => (vec![-1; n], 0, 1, false, false),
    };
    let cfg = BifactorMultigroupConfig {
        q_general: r.q_general,
        q_specific: qs,
        max_iter: r.max_iter,
        tol: r.tol,
        n_starts: r.n_starts,
        seed: r.seed,
        newton_iter: r.newton_iter,
        ridge: r.ridge,
        estimate_specific_means: means,
        estimate_specific_vars: vars,
        device,
        slope_prior: crate::bifactor_grm::SlopePrior::None,
    };
    MaskFit::fit_bifactor(
        &r.y,
        r.observed.as_deref(),
        &r.group_id,
        r.n_groups,
        n,
        r.n_cat,
        &r.anchor,
        &cfg,
        &map,
        ns,
    )?
    .export_private()
}
