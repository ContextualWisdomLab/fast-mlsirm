//! Native typed-state recovery and missing-only lifecycle scheduling.
//! This consumer retains the existing LR, identification, Holm and convergence rules.
//! Trusted private files bind exact returned fits; they do not authenticate execution.
use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
#[cfg(target_os = "macos")]
fn nofollow() -> i32 {
    0x100
}
#[cfg(target_os = "linux")]
fn nofollow() -> i32 {
    0x20000
}
fn read(path: &Path, cap: u64) -> Result<String, String> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    // O_NOFOLLOW on the supported macOS/Unix private consumer.
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(nofollow())
        .open(path)
        .map_err(|e| e.to_string())?;
    let m = f.metadata().map_err(|e| e.to_string())?;
    if !m.is_file() || m.len() > cap {
        return Err("private input size/type refused".into());
    }
    let mut raw = String::new();
    f.by_ref()
        .take(cap + 1)
        .read_to_string(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() as u64 > cap {
        return Err("private input exceeds cap".into());
    }
    Ok(raw)
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRef {
    path: PathBuf,
    sha256: String,
}
#[path = "mask_grm_resume_sources.rs"]
mod source_binding;
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    schema: u32,
    request: Value,
    journal: FileRef,
    states: Vec<FileRef>,
    compiled_sources: Value,
    provenance: Vec<FileRef>,
    checkpoint_authority: FileRef,
}
fn pinned_spec(path: &Path, expected: &str) -> Result<String, String> {
    let raw = read(path, 32 * 1024 * 1024)?;
    if expected.len() != 64 || sha(raw.as_bytes()) != expected {
        return Err("external resume spec digest mismatch".into());
    }
    Ok(raw)
}
fn checked(ref_: &FileRef, cap: u64) -> Result<String, String> {
    let raw = read(&ref_.path, cap)?;
    if sha(raw.as_bytes()) != ref_.sha256 {
        return Err("private input digest mismatch".into());
    }
    Ok(raw)
}

/// Return uncompleted item indices. A retained failure is not a completed pair.
pub fn remaining_items(n: usize, pairs: &[Value]) -> Result<Vec<usize>, String> {
    if n == 0 {
        return Err("empty family".into());
    }
    let mut seen = HashSet::new();
    for pair in pairs {
        let i = pair["tested_item"].as_u64().ok_or("missing tested item")? as usize;
        let p = pair["p_value"].as_f64().ok_or("incomplete pair")?;
        if i >= n
            || !seen.insert(i)
            || pair["status"] != "complete"
            || !p.is_finite()
            || !(0.0..=1.0).contains(&p)
        {
            return Err("invalid or duplicate completed pair".into());
        }
    }
    Ok((0..n).filter(|i| !seen.contains(i)).collect())
}
fn missing_with<T>(
    n: usize,
    pairs: &[Value],
    item: usize,
    run: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    if remaining_items(n, pairs)? != vec![item] {
        return Err("completed or non-exact missing unit refused".into());
    }
    run()
}
/// Pure family policy, only called on a complete uniquely indexed admitted family.
pub fn complete_family(anchors: &[bool], pairs: &[Value], alpha: f64) -> Result<Value, String> {
    if !remaining_items(anchors.len(), pairs)?.is_empty() {
        return Err("family has unfinished pairs".into());
    }
    let mut ordered = pairs.to_vec();
    ordered.sort_by_key(|p| p["tested_item"].as_u64().unwrap());
    let raw: Vec<_> = ordered
        .iter()
        .map(|p| p["p_value"].as_f64().unwrap())
        .collect();
    let (adj, removed) = select_removal(anchors, &raw, alpha)?;
    let mut final_anchor = anchors.to_vec();
    if let Some(j) = removed {
        final_anchor[j] = false;
    }
    Ok(
        json!({"anchors_before":anchors,"item_results":ordered,"adjusted_p":adj,"removed":removed,
        "final_anchor":final_anchor,"status":if removed.is_some(){"round_limit_incomplete"}else{"stable_no_further_removal"}}),
    )
}
fn bind_pair_summaries(pair: &Value, null: &Value, alt: &Value) -> Result<(), String> {
    if pair["null"] != *null || pair["alternative"] != *alt {
        return Err("retained pair/native summary mismatch".into());
    }
    Ok(())
}
fn config(r: &Request) -> BifactorMultigroupConfig {
    BifactorMultigroupConfig {
        q_general: r.q_general,
        q_specific: r.bifactor.as_ref().map_or(1, |b| b.q_specific),
        max_iter: r.max_iter,
        tol: r.tol,
        n_starts: r.n_starts,
        seed: r.seed,
        newton_iter: r.newton_iter,
        ridge: r.ridge,
        estimate_specific_means: r
            .bifactor
            .as_ref()
            .is_some_and(|b| b.estimate_specific_means),
        estimate_specific_vars: r
            .bifactor
            .as_ref()
            .is_some_and(|b| b.estimate_specific_vars),
        device: crate::Device::Gpu,
        slope_prior: crate::bifactor_grm::SlopePrior::None,
    }
}
fn matches_request(f: &MaskFit, r: &Request) -> Result<(), String> {
    let map = r
        .bifactor
        .as_ref()
        .map_or_else(|| vec![-1; r.item_ids.len()], |b| b.specific_map.clone());
    if f.y != r.y
        || f.observed != r.observed
        || f.groups != r.group_id
        || f.n_groups != r.n_groups
        || f.n_items != r.item_ids.len()
        || f.n_cat != r.n_cat
        || f.specific_map != map
        || f.n_specific != r.bifactor.as_ref().map_or(0, |b| b.n_specific)
        || format!("{:?}", f.cfg) != format!("{:?}", config(r))
    {
        return Err("fit/request identity mismatch".into());
    }
    if !f.result.converged || f.result.termination_reason != "tolerance_met" {
        return Err("retained fit not converged".into());
    }
    f.validate_gpu_evidence()
}
fn restore(raw: &str) -> Result<MaskFit, String> {
    let f = MaskFit::restore_private(raw)?;
    if f.export_private()? != raw {
        return Err("typed state roundtrip mismatch".into());
    }
    Ok(f)
}

struct Recovered {
    spec_sha256: String,
    request: Request,
    fits: HashMap<String, MaskFit>,
    initial_id: String,
    pairs: Vec<Value>,
    state_refs: HashMap<String, Value>,
}
impl Recovered {
    fn load(path: &Path, expected: &str) -> Result<Self, String> {
        let raw = pinned_spec(path, expected)?;
        let spec: Spec = serde_json::from_str(&raw).map_err(|_| "invalid resume spec")?;
        if spec.compiled_sources != source_binding::current() {
            return Err("whole compiled source/dependency binding mismatch".into());
        }
        if spec.provenance.is_empty() {
            return Err("missing reviewed provenance".into());
        }
        for p in &spec.provenance {
            checked(p, 32 * 1024 * 1024)?;
        }
        let text = spec.request.to_string();
        let r: Request = serde_json::from_str(&text).map_err(|_| "invalid resume request")?;
        if spec.schema != 1
            || r.device != "gpu"
            || !r.controls.require_gpu
            || r.studied_item.is_some()
            || r.max_rounds != 1
            || !r.alpha.is_finite()
            || r.alpha <= 0.0
            || r.alpha >= 1.0
            || r.item_ids.is_empty()
            || r.item_ids.len() != r.anchor.len()
            || r.item_ids.iter().any(|s| s.is_empty())
            || r.item_ids.iter().collect::<HashSet<_>>().len() != r.item_ids.len()
            || !r.anchor.iter().all(|a| *a)
        {
            return Err("resume requires the explicit all-anchor single-round GPU family".into());
        }
        let authority: Value =
            serde_json::from_str(&checked(&spec.checkpoint_authority, 1024 * 1024)?)
                .map_err(|_| "invalid checkpoint authority")?;
        let authentic = authority["authentic_states"]
            .as_array()
            .ok_or("missing authentic state set")?;
        let claimed: HashSet<_> = spec.states.iter().map(|v| v.sha256.as_str()).collect();
        let qualified: HashSet<_> = authentic
            .iter()
            .filter_map(|v| v["sha256"].as_str())
            .collect();
        if claimed != qualified
            || claimed.len() != 4
            || spec.states.len() != 4
            || spec.journal.sha256 != authority["input_artifacts"]["fresh_full/stderr"]
            || json!(r.item_ids) != authority["model"]["item_ids"]
            || json!(r.anchor) != authority["model"]["null_anchor"]
            || json!(r.n_cat) != authority["model"]["n_cat"]
            || json!(r.n_groups) != authority["model"]["n_groups"]
            || json!(r.group_id.len()) != authority["model"]["n_persons"]
            || authority["model"]["reference_group"] != 0
            || serde_json::to_value(&r.controls).map_err(|e| e.to_string())?
                != authority["inference_controls"]
            || json!(r.alpha) != authority["family_policy"]["alpha"]
            || json!(r.max_rounds) != authority["family_policy"]["max_rounds"]
        {
            return Err("external request/journal/authentic state authority mismatch".into());
        }
        let cfg = serde_json::to_value(config(&r)).map_err(|e| e.to_string())?;
        if cfg
            .as_object()
            .ok_or("invalid config")?
            .iter()
            .any(|(k, v)| authority["solver"][k] != *v)
            || authority["solver"]["internal_gradient_tolerance"] != 1e-9
            || authority["solver"]["internal_step_tolerance"] != 1e-9
        {
            return Err("external solver authority mismatch".into());
        }
        let journal = checked(&spec.journal, 64 * 1024 * 1024)?;
        let mut initial = None;
        let mut pairs = Vec::new();
        let mut begin = false;
        let mut round = false;
        for line in journal.lines() {
            let Some(text) = line.strip_prefix("FMLSIRM_DIF_PROGRESS ") else {
                return Err("unexpected journal bytes".into());
            };
            if text.len() > 8 * 1024 * 1024 {
                return Err("journal record exceeds cap".into());
            }
            let e: Value = serde_json::from_str(text).map_err(|_| "invalid journal record")?;
            match e["event"].as_str().ok_or("missing journal event")? {
                "lifecycle_initial_start" => {
                    if begin || e["request_sha256"] != sha(spec.request.to_string().as_bytes()) {
                        return Err("initial request digest mismatch/duplicate".into());
                    }
                    begin = true;
                }
                "lifecycle_initial_terminal" => {
                    if !begin || initial.is_some() || round {
                        return Err("initial journal order".into());
                    }
                    initial = Some(e["result"]["fit"].clone());
                }
                "round_start" => {
                    if initial.is_none()
                        || round
                        || e["round"] != 1
                        || e["anchors"] != json!(r.anchor)
                        || e["family_size"] != r.item_ids.len()
                    {
                        return Err("round identity mismatch".into());
                    }
                    round = true;
                }
                "pair_complete" => {
                    if !round
                        || e["family_size"] != r.item_ids.len()
                        || e["round"] != 1
                        || e["tested_item"] != e["result"]["tested_item"]
                        || e["tested_item"].is_null()
                    {
                        return Err("pair journal identity mismatch".into());
                    }
                    pairs.push(e["result"].clone());
                }
                "fit_start" | "optimizer_start" | "optimizer_terminal" => {
                    if !begin {
                        return Err("optimizer journal precedes request".into());
                    }
                }
                _ => return Err("unexpected completed/unknown lifecycle event".into()),
            }
        }
        if !round || !journal.ends_with('\n') {
            return Err("incomplete journal boundary".into());
        }
        if pairs
            .iter()
            .enumerate()
            .any(|(i, p)| p["tested_item"].as_u64() != Some(i as u64))
        {
            return Err("completed journal item order mismatch".into());
        }
        remaining_items(r.item_ids.len(), &pairs)?;
        let initial = initial.ok_or("missing authentic initial summary")?;
        let initial_id = initial["fit_identity"]
            .as_str()
            .ok_or("missing initial identity")?
            .to_owned();
        let mut fits = HashMap::new();
        let mut state_refs = HashMap::new();
        if spec.states.len() != pairs.len() + 1 {
            return Err("saved state coverage mismatch".into());
        }
        for file in &spec.states {
            let f = restore(&checked(file, 16 * 1024 * 1024)?)?;
            matches_request(&f, &r)?;
            let id = f.identity()?;
            let qualified = authentic
                .iter()
                .find(|v| v["sha256"] == file.sha256)
                .ok_or("unqualified state")?;
            if qualified["identity"] != id
                || qualified["anchor"] != json!(f.anchor)
                || qualified["embedded_parent"] != json!(f.embedded_parent)
                || qualified["native_likelihood_bits"].as_u64()
                    != Some(f.loglik_at(&f.packed()?)?.to_bits())
                || qualified["trace_bits"].as_u64() != Some(f.trace_loglik().to_bits())
            {
                return Err("authentic state/native readback mismatch".into());
            }
            if fits.insert(id.clone(), f).is_some() {
                return Err("duplicate fit identity".into());
            }
            state_refs.insert(id, json!({"path":file.path,"sha256":file.sha256}));
        }
        let null = fits.get(&initial_id).ok_or("authentic null state absent")?;
        if null.anchor != r.anchor || null.summary() != Ok(initial.clone()) {
            return Err("initial typed summary mismatch".into());
        }
        let mut used = HashSet::from([initial_id.clone()]);
        for p in &pairs {
            let id = p["alternative"]["fit_identity"]
                .as_str()
                .ok_or("missing alternative identity")?;
            let alt = fits.get(id).ok_or("authentic alternative state absent")?;
            validate_pair(p, null, alt, &r)?;
            used.insert(id.to_owned());
        }
        if used.len() != fits.len() {
            return Err("unassociated saved state".into());
        }
        if remaining_items(r.item_ids.len(), &pairs)?.len() != 1 {
            return Err("consumer requires exactly one missing unit".into());
        }
        Ok(Self {
            spec_sha256: sha(raw.as_bytes()),
            request: r,
            fits,
            initial_id,
            pairs,
            state_refs,
        })
    }
    fn null(&self) -> &MaskFit {
        &self.fits[&self.initial_id]
    }
    fn inspect(&self) -> Result<Value, String> {
        let mut completed=self.pairs.iter().map(|p|json!({"item_index":p["tested_item"],"item_id":self.request.item_ids[p["tested_item"].as_u64().unwrap() as usize],
        "null_identity":p["null"]["fit_identity"],"alternative_identity":p["alternative"]["fit_identity"],"mask":p["alternative"]["anchor"]})).collect::<Vec<_>>();
        completed.sort_by_key(|p| p["item_index"].as_u64().unwrap());
        Ok(
            json!({"schema":1,"spec_sha256":self.spec_sha256,"item_ids":self.request.item_ids,"anchor":self.request.anchor,
        "initial_identity":self.initial_id,"states":self.state_refs,"completed":completed,"remaining_items":remaining_items(self.request.item_ids.len(),&self.pairs)?,
        "initial_fit_launches":0,"completed_pair_refits":0,"fit_launches":0,"family_adjustment_complete":false,"selected_final_complete":false,
        "native_state_readbacks":self.fits.iter().map(|(id,f)|Ok(json!({"fit_identity":id,
            "native_likelihood_bits":f.loglik_at(&f.packed()?)?.to_bits(),"trace_bits":f.trace_loglik().to_bits(),
            "anchor":f.anchor,"converged":f.result.converged,"byte_identical":true}))).collect::<Result<Vec<Value>,String>>()?}),
        )
    }
    fn alt_from_file(&self, path: &Path, item: usize) -> Result<MaskFit, String> {
        let f = restore(&read(path, 16 * 1024 * 1024)?)?;
        matches_request(&f, &self.request)?;
        let mut expected = self.request.anchor.clone();
        expected[item] = false;
        if f.anchor != expected || f.embedded_parent.as_deref() != Some(self.initial_id.as_str()) {
            return Err("alternative mask/parent mismatch".into());
        }
        Ok(f)
    }
}
fn validate_pair(p: &Value, null: &MaskFit, alt: &MaskFit, r: &Request) -> Result<(), String> {
    let j = p["tested_item"].as_u64().ok_or("missing pair item")? as usize;
    remaining_items(r.item_ids.len(), std::slice::from_ref(p))?;
    let mut mask = r.anchor.clone();
    mask[j] = false;
    let ll0 = null.loglik_at(&null.packed()?)?;
    let ll1 = alt.loglik_at(&alt.packed()?)?;
    let raw = 2.0 * (ll1 - ll0);
    let backend = 2.0 * (alt.trace_loglik() - null.trace_loglik());
    if null.anchor != r.anchor
        || alt.anchor != mask
        || alt.embedded_parent.as_deref() != Some(null.identity()?.as_str())
        || p["controls"] != serde_json::to_value(&r.controls).map_err(|e| e.to_string())?
        || p["df"].as_u64()
            != Some(((r.n_groups - 1) * (r.n_cat + usize::from(alt.specific_map[j] >= 0))) as u64)
        || p["information_device"] != "cpu_f64"
        || p["inference_scope"] != "asymptotic_conditional_on_current_anchors"
        || raw < 0.0
        || backend < 0.0
        || (raw - backend).abs() > r.controls.lr_backend_tolerance
        || (ll0 - null.trace_loglik()).abs() > r.controls.fit_ll_tolerance
        || (ll1 - alt.trace_loglik()).abs() > r.controls.fit_ll_tolerance
        || alt
            .embedded_start
            .is_none_or(|v| (v - null.trace_loglik()).abs() > r.controls.fit_ll_tolerance)
        || p["embedding_loglik"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || (v - ll0).abs() > r.controls.ll_roundoff_tolerance)
    {
        return Err("retained pair contract mismatch".into());
    }
    bind_pair_summaries(p, &null.summary()?, &alt.summary()?)?;
    let df = (r.n_groups - 1) * (r.n_cat + usize::from(alt.specific_map[j] >= 0));
    if p["raw_lr_cpu_f64"].as_f64().map(f64::to_bits) != Some(raw.to_bits())
        || p["raw_lr_fit_backend"].as_f64().map(f64::to_bits) != Some(backend.to_bits())
        || p["p_value"].as_f64().map(f64::to_bits)
            != Some(crate::fitstats::chi2_sf(raw, df as f64).to_bits())
    {
        return Err("retained LR/p-value mismatch".into());
    }
    if p["effects"] != "not_computed" {
        return Err("unexpected retained effect payload".into());
    }
    let infos = p["information"]
        .as_array()
        .ok_or("missing retained information")?;
    if infos.len() != 4 {
        return Err("retained information must have all four evaluations".into());
    }
    for (k, info) in infos.iter().enumerate() {
        let f = if k < 2 { null } else { alt };
        let d = f.packed()?.len();
        if info["fit_identity"] != f.identity()?
            || info["dimension"] != d
            || info["rank"] != d
            || info["negative_eigenvalues"] != 0
            || info["matrix"].as_array().is_none_or(|a| {
                a.len() != d * d || a.iter().any(|v| v.as_f64().is_none_or(|x| !x.is_finite()))
            })
            || info["eigenvalues"].as_array().is_none_or(|a| {
                a.len() != d || a.iter().any(|v| v.as_f64().is_none_or(|x| !x.is_finite()))
            })
            || info["step"].as_f64()
                != Some(if k % 2 == 0 {
                    r.controls.fd_step
                } else {
                    r.controls.fd_step * 0.5
                })
            || info["rank_atol"].as_f64() != Some(r.controls.rank_atol)
            || info["rank_rtol"].as_f64() != Some(r.controls.rank_rtol)
            || info["objective_calls"] != 2 * d * d + 1
            || info["min_eigenvalue"]
                .as_f64()
                .zip(info["threshold"].as_f64())
                .is_none_or(|(v, t)| !v.is_finite() || !t.is_finite() || v <= t)
        {
            return Err("retained information association mismatch".into());
        }
    }
    for (a, b) in [(0, 1), (2, 3)] {
        let m = infos[a]["matrix"].as_array().unwrap();
        let n = infos[b]["matrix"].as_array().unwrap();
        let scale = m
            .iter()
            .map(|v| v.as_f64().unwrap().abs())
            .fold(0.0, f64::max);
        if m.iter().zip(n).any(|(x, y)| {
            (x.as_f64().unwrap() - y.as_f64().unwrap()).abs()
                > r.controls.rank_atol + r.controls.hessian_rtol * scale
        }) {
            return Err("retained information step instability".into());
        }
    }
    Ok(())
}

/// Nonreplacing durable publication in a caller-owned new private directory.
fn publish(path: &Path, raw: &[u8]) -> Result<(), String> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let parent = path.parent().ok_or("missing output directory")?;
    let tmp = path.with_extension("pending");
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)
        .map_err(|e| e.to_string())?;
    f.write_all(raw)
        .and_then(|_| f.sync_all())
        .map_err(|e| e.to_string())?;
    std::fs::hard_link(&tmp, path).map_err(|e| e.to_string())?;
    std::fs::remove_file(tmp).map_err(|e| e.to_string())?;
    std::fs::File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())
}
fn new_dir(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(path)
        .map_err(|e| e.to_string())?;
    std::fs::File::open(path.parent().ok_or("missing parent")?)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())
}
fn gpu_lease() -> Result<(), String> {
    if std::env::var("FMLS_DS_GPU_LEASED").as_deref() != Ok("1") {
        return Err("parent GPU admission/lease required".into());
    }
    Ok(())
}
fn merged_family(rec: &Recovered, pair_file: &Path, alt_file: &Path) -> Result<Value, String> {
    let p: Value = serde_json::from_str(&read(pair_file, 16 * 1024 * 1024)?)
        .map_err(|_| "invalid pair artifact")?;
    if p["spec_sha256"] != rec.spec_sha256 {
        return Err("pair/spec mismatch".into());
    }
    let item = p["result"]["tested_item"]
        .as_u64()
        .ok_or("pair item missing")? as usize;
    let left = remaining_items(rec.request.item_ids.len(), &rec.pairs)?;
    if left != vec![item] {
        return Err("additional pair is not exact remaining unit".into());
    }
    let alt = rec.alt_from_file(alt_file, item)?;
    if p["alternative_sha256"] != sha(read(alt_file, 16 * 1024 * 1024)?.as_bytes()) {
        return Err("pair/alternative digest mismatch".into());
    }
    validate_pair(&p["result"], rec.null(), &alt, &rec.request)?;
    let mut pairs = rec.pairs.clone();
    pairs.push(p["result"].clone());
    complete_family(&rec.request.anchor, &pairs, rec.request.alpha)
}

/// CLI: MODE SPEC EXPECTED_SPEC_SHA256 [mode arguments].
/// Disabled by default. The parent owns review/admission, resources and settlement;
/// no scientific elapsed-time limit is imposed by this consumer.
pub fn main_cli() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mode = a.first().ok_or("mode required")?;
    let expected = match mode.as_str() {
        "inspect" => 3,
        "alternative" => 5,
        "pair" => 6,
        "family" => 6,
        "final" => 7,
        _ => return Err("unknown mode".into()),
    };
    if a.len() != expected {
        return Err("wrong argument count".into());
    }
    let rec = Recovered::load(Path::new(&a[1]), &a[2])?;
    if mode == "inspect" {
        println!("{}", rec.inspect()?);
        return Ok(());
    }
    if std::env::var("FMLS_RESUME_ADMITTED_SPEC").as_deref() != Ok(rec.spec_sha256.as_str())
        || std::env::var("FMLS_RESUME_ADMITTED_UNIT").as_deref() != Ok(mode.as_str())
    {
        return Err("disabled: separate parent unit admission required".into());
    }
    for (index, name) in match mode.as_str() {
        "pair" => vec![(4, "ALTERNATIVE")],
        "family" => vec![(3, "PAIR"), (4, "ALTERNATIVE")],
        "final" => vec![(3, "PAIR"), (4, "ALTERNATIVE"), (5, "FAMILY")],
        _ => vec![],
    } {
        let expected = std::env::var(format!("FMLS_RESUME_ADMITTED_{name}_SHA256"))
            .map_err(|_| "missing parent artifact digest")?;
        if sha(read(Path::new(&a[index]), 32 * 1024 * 1024)?.as_bytes()) != expected {
            return Err("parent artifact digest mismatch".into());
        }
    }
    let out = Path::new(a.last().unwrap());
    progress_event(
        json!({"event":"resume_unit_start","unit":mode,"spec_sha256":rec.spec_sha256,"fit_launches":0}),
    )?;
    let value = match mode.as_str() {
        "alternative" | "pair" => {
            let item: usize = a[3].parse().map_err(|_| "invalid item")?;
            if remaining_items(rec.request.item_ids.len(), &rec.pairs)? != vec![item] {
                return Err("completed or non-exact missing unit refused".into());
            }
            if mode == "alternative" {
                gpu_lease()?;
                new_dir(out)?;
                // The sole fit call in this consumer releases only the missing item.
                let alt = missing_with(rec.request.item_ids.len(), &rec.pairs, item, || {
                    state::with_private_checkpoints(&out.join("typed"), || {
                        rec.null().release_item(item)
                    })
                })?;
                let raw = alt.export_private()?;
                let digest = sha(raw.as_bytes());
                json!({"spec_sha256":rec.spec_sha256,"tested_item":item,"fit":alt.summary()?,
                "state":out.join("typed").join(format!("{digest}.maskfit.json")),"sha256":digest,"fit_launches":1})
            } else {
                let alt = rec.alt_from_file(Path::new(&a[4]), item)?;
                let result = missing_with(rec.request.item_ids.len(), &rec.pairs, item, || {
                    rec.null().nested_lr(&alt, item, &rec.request.controls)
                })?;
                new_dir(out)?;
                json!({"spec_sha256":rec.spec_sha256,"alternative_sha256":sha(read(Path::new(&a[4]),16*1024*1024)?.as_bytes()),"result":result,"fit_launches":0})
            }
        }
        "family" => {
            let family = merged_family(&rec, Path::new(&a[3]), Path::new(&a[4]))?;
            new_dir(out)?;
            json!({"spec_sha256":rec.spec_sha256,"family":family,"fit_launches":0})
        }
        "final" => {
            let family = merged_family(&rec, Path::new(&a[3]), Path::new(&a[4]))?;
            let saved: Value = serde_json::from_str(&read(Path::new(&a[5]), 32 * 1024 * 1024)?)
                .map_err(|_| "invalid family artifact")?;
            if saved["spec_sha256"] != rec.spec_sha256
                || saved["family"] != family
                || family["status"] != "stable_no_further_removal"
                || family["final_anchor"] != json!(rec.request.anchor)
            {
                return Err("selected final requires a stable complete family".into());
            }
            // Identical request/mask: select the authentic completed fit; no initial refit.
            let f = rec.null();
            f.validate_gpu_evidence()?;
            new_dir(out)?;
            let raw = f.export_private()?;
            publish(&out.join("selected.maskfit.json"), raw.as_bytes())?;
            json!({"spec_sha256":rec.spec_sha256,"status":"complete_gpu_fit_host_information","full_gpu":false,
            "selected_fit":f.summary()?,"selected_state_sha256":sha(raw.as_bytes()),"fit_launches":0,"selection":"exact_existing_native_fit"})
        }
        _ => unreachable!(),
    };
    publish(&out.join("result.json"), value.to_string().as_bytes())?;
    progress_event(
        json!({"event":"resume_unit_terminal","unit":mode,"spec_sha256":rec.spec_sha256,"fit_launches":value["fit_launches"]}),
    )?;
    println!(
        "{}",
        json!({"mode":mode,"result":out.join("result.json"),"sha256":sha(value.to_string().as_bytes())})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pairs(n: usize) -> Vec<Value> {
        (0..n)
            .map(|i| json!({"tested_item":i,"status":"complete","p_value":0.5}))
            .collect()
    }
    #[test]
    fn complete_family_refuses_partial_adjustment() {
        assert!(complete_family(&[true; 4], &pairs(3), 0.05).is_err());
    }
    #[test]
    fn retained_pair_binds_complete_native_summaries() {
        let null = json!({"fit_identity":"n","anchor":[true,true],"packed":[1.0]});
        let alt = json!({"fit_identity":"a","anchor":[false,true],"packed":[2.0]});
        let mut p = json!({"tested_item":0,"null":null,"alternative":alt});
        p["alternative"]["packed"][0] = json!(3.0);
        assert!(bind_pair_summaries(&p, &null, &alt).is_err());
    }
    #[test]
    fn missing_gate_calls_only_missing_and_preserves_completed_pairs() {
        let calls = std::cell::Cell::new(0);
        let p = pairs(3);
        let original = p.clone();
        for i in 0..3 {
            assert!(missing_with(4, &p, i, || {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .is_err());
        }
        assert_eq!(calls.get(), 0);
        missing_with(4, &p, 3, || {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(p, original);
    }
    #[test]
    fn null_p_and_malformed_indices_refuse() {
        let mut p = pairs(3);
        p[0]["p_value"] = Value::Null;
        assert!(remaining_items(4, &p).is_err());
        p = pairs(3);
        p[0]["tested_item"] = json!(-1);
        assert!(remaining_items(4, &p).is_err());
    }
    #[test]
    fn exact_remaining_does_not_repeat_completed_units() {
        assert_eq!(remaining_items(4, &pairs(3)).unwrap(), vec![3]);
    }
    #[test]
    fn duplicate_out_of_range_and_failed_pairs_refuse() {
        let mut p = pairs(3);
        p.push(p[0].clone());
        assert!(remaining_items(4, &p).is_err());
        p = pairs(3);
        p[0]["tested_item"] = json!(4);
        assert!(remaining_items(4, &p).is_err());
        p = pairs(3);
        p[0]["status"] = json!("nonconverged");
        assert!(remaining_items(4, &p).is_err());
    }
    #[test]
    fn full_family_order_is_original_item_order() {
        let mut p = pairs(4);
        p.reverse();
        let a = complete_family(&[true; 4], &p, 0.05).unwrap();
        assert_eq!(a["item_results"][0]["tested_item"], 0);
        assert_eq!(a["adjusted_p"], json!([1.0, 1.0, 1.0, 1.0]));
        assert_eq!(a["status"], "stable_no_further_removal");
    }
    #[test]
    #[ignore = "explicit authentic synthetic restore-only controls; no optimizer"]
    fn authentic_controls_restore_pairs_and_resealed_contradictions() {
        let path = std::env::var("FMLS_RESUME_TEST_SPEC").expect("explicit private spec");
        let hash = std::env::var("FMLS_RESUME_TEST_SHA256").expect("external expected digest");
        let rec = Recovered::load(Path::new(&path), &hash).unwrap();
        assert_eq!(
            remaining_items(rec.request.item_ids.len(), &rec.pairs).unwrap(),
            vec![3]
        );
        for p in &rec.pairs {
            let alt = &rec.fits[p["alternative"]["fit_identity"].as_str().unwrap()];
            validate_pair(p, rec.null(), alt, &rec.request).unwrap();
            for field in [
                "packed",
                "starts",
                "execution_evidence",
                "dispatch",
                "loglik_trace",
            ] {
                let mut wrong = p.clone();
                wrong["alternative"][field] = Value::Null;
                assert!(
                    validate_pair(&wrong, rec.null(), alt, &rec.request).is_err(),
                    "{field}"
                );
            }
            for key in [
                "information",
                "p_value",
                "raw_lr_cpu_f64",
                "embedding_loglik",
                "controls",
            ] {
                let mut wrong = p.clone();
                wrong[key] = Value::Null;
                assert!(
                    validate_pair(&wrong, rec.null(), alt, &rec.request).is_err(),
                    "{key}"
                );
            }
        }
    }
    #[test]
    #[ignore = "explicit authentic synthetic restore-only request controls; no optimizer"]
    fn authentic_controls_request_mismatches_before_missing_call() {
        let path = std::env::var("FMLS_RESUME_TEST_SPEC").unwrap();
        let hash = std::env::var("FMLS_RESUME_TEST_SHA256").unwrap();
        let rec = Recovered::load(Path::new(&path), &hash).unwrap();
        let spec: Value =
            serde_json::from_str(&read(Path::new(&path), 32 * 1024 * 1024).unwrap()).unwrap();
        let cases = [
            ("y", json!(vec![0; rec.request.y.len()])),
            ("observed", json!(vec![true; rec.request.y.len()])),
            ("group_id", json!(vec![1; rec.request.group_id.len()])),
            ("n_cat", json!(3)),
            ("n_groups", json!(3)),
            ("q_general", json!(9)),
            ("max_iter", json!(101)),
            ("tol", json!(1e-7)),
            ("n_starts", json!(2)),
            ("seed", json!(43)),
            ("newton_iter", json!(21)),
            ("ridge", json!(1e-5)),
            (
                "bifactor",
                json!({"specific_map":[0,0,0,0],"n_specific":1,"q_specific":3,"estimate_specific_means":true,"estimate_specific_vars":true}),
            ),
        ];
        let calls = std::cell::Cell::new(0);
        for (key, value) in cases {
            let mut v = spec["request"].clone();
            v[key] = value;
            let r: Request = serde_json::from_value(v).unwrap();
            let run = || {
                matches_request(rec.null(), &r)?;
                missing_with(4, &rec.pairs, 3, || {
                    calls.set(calls.get() + 1);
                    Ok(())
                })
            };
            assert!(run().is_err(), "{key}");
            assert_eq!(calls.get(), 0);
        }
    }
    #[test]
    fn single_removal_requires_another_round() {
        let mut p = pairs(4);
        p[3]["p_value"] = json!(0.001);
        let a = complete_family(&[true; 4], &p, 0.05).unwrap();
        assert_eq!(a["removed"], 3);
        assert_eq!(a["status"], "round_limit_incomplete");
    }
}
