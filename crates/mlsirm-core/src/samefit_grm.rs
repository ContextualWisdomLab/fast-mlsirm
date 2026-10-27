//! Packed general-only GRM coordinates; no optimizer or inference admission.
//! Moment consumers follow Maydeu-Olivares (2013), equations 4, 10–11.
//! Reference: Maydeu-Olivares, A. (2013). Goodness-of-fit assessment of item
//! response theory models. Measurement, 11, 71–101. doi:10.1080/15366367.2013.831680.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Coordinate {
    Slope(usize, usize),
    FirstIntercept(usize, usize),
    LogGap(usize, usize, usize),
    Mean(usize),
    LogSd(usize),
}

/// Bit-preserving coordinate archive, NOT a verified historical fit archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedArchive {
    schema_version: u32,
    anchor: Vec<bool>,
    groups: usize,
    categories: usize,
    packed_bits: Vec<u64>,
}
impl PackedArchive {
    /// Preserve binary64 input coordinates and validate their decoded geometry.
    pub fn new(
        anchor: Vec<bool>,
        groups: usize,
        categories: usize,
        packed: Vec<f64>,
    ) -> Result<Self, String> {
        let a = Self {
            schema_version: 1,
            anchor,
            groups,
            categories,
            packed_bits: packed.into_iter().map(f64::to_bits).collect(),
        };
        decode(&a)?;
        Ok(a)
    }
    pub fn packed_bits(&self) -> &[u64] {
        &self.packed_bits
    }
    pub fn n_groups(&self) -> usize {
        self.groups
    }
    pub fn n_items(&self) -> usize {
        self.anchor.len()
    }
    pub fn n_categories(&self) -> usize {
        self.categories
    }
    /// Deterministic historical mask coordinate order, reference group zero.
    pub fn coordinates(&self) -> Result<Vec<Coordinate>, String> {
        if self.schema_version != 1
            || self.anchor.is_empty()
            || self.groups < 2
            || self.categories < 2
        {
            return Err("invalid archive geometry/schema".into());
        }
        let count = self
            .anchor
            .iter()
            .try_fold(0usize, |n, &a| {
                n.checked_add(if a { 1 } else { self.groups })
            })
            .and_then(|n| n.checked_mul(self.categories))
            .and_then(|n| {
                self.groups
                    .checked_sub(1)
                    .and_then(|g| g.checked_mul(2))
                    .and_then(|g| n.checked_add(g))
            })
            .ok_or("coordinate size overflow")?;
        if count != self.packed_bits.len() {
            return Err("coordinate length does not match anchor geometry".into());
        }
        let mut c = Vec::with_capacity(count);
        for (i, &a) in self.anchor.iter().enumerate() {
            for g in 0..if a { 1 } else { self.groups } {
                c.push(Coordinate::Slope(g, i));
                c.push(Coordinate::FirstIntercept(g, i));
                for k in 1..self.categories - 1 {
                    c.push(Coordinate::LogGap(g, i, k));
                }
            }
        }
        for g in 1..self.groups {
            c.push(Coordinate::Mean(g));
            c.push(Coordinate::LogSd(g));
        }
        Ok(c)
    }
    pub fn to_json(&self) -> Result<String, String> {
        decode(self)?;
        serde_json::to_string(self).map_err(|e| e.to_string())
    }
    pub fn from_json(s: &str) -> Result<Self, String> {
        let a: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        decode(&a)?;
        Ok(a)
    }
}
#[derive(Debug, Serialize)]
pub struct Decoded {
    pub slope: Vec<f64>,
    pub intercepts: Vec<f64>,
    pub means: Vec<f64>,
    pub sds: Vec<f64>,
}

/// Decode without fitting; transformed dense bits are not historical dense bits.
pub fn decode(a: &PackedArchive) -> Result<Decoded, String> {
    let coords = a.coordinates()?;
    let j = a.anchor.len();
    let z = a.categories - 1;
    let size = a.groups.checked_mul(j).ok_or("dense size overflow")?;
    let cells = size.checked_mul(z).ok_or("dense size overflow")?;
    if cells > 16_000_000 {
        return Err("dense workspace exceeds 16 million elements".into());
    }
    let mut d = Decoded {
        slope: vec![0.0; size],
        intercepts: vec![0.0; cells],
        means: vec![0.0; a.groups],
        sds: vec![1.0; a.groups],
    };
    for (c, &bits) in coords.iter().zip(&a.packed_bits) {
        let v = f64::from_bits(bits);
        if !v.is_finite() {
            return Err("nonfinite coordinate".into());
        }
        match *c {
            Coordinate::Slope(g, i) => d.slope[g * j + i] = v,
            Coordinate::FirstIntercept(g, i) => d.intercepts[(g * j + i) * z] = v,
            Coordinate::LogGap(g, i, k) => {
                let n = (g * j + i) * z + k;
                d.intercepts[n] = d.intercepts[n - 1] - v.exp();
            }
            Coordinate::Mean(g) => d.means[g] = v,
            Coordinate::LogSd(g) => d.sds[g] = v.exp(),
        }
    }
    for g in 1..a.groups {
        for (i, &common) in a.anchor.iter().enumerate() {
            if common {
                d.slope[g * j + i] = d.slope[i];
                for k in 0..z {
                    d.intercepts[(g * j + i) * z + k] = d.intercepts[i * z + k];
                }
            }
        }
    }
    if d.sds.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || d.intercepts.iter().any(|v| !v.is_finite())
        || d.intercepts
            .chunks_exact(z)
            .any(|v| v.windows(2).any(|w| w[0] <= w[1]))
    {
        return Err("coordinate transform overflow/underflow".into());
    }
    Ok(d)
}

/// Bit-preserving multigroup bifactor GRM coordinate archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BifactorPackedArchive {
    schema_version: u32,
    anchor: Vec<bool>,
    groups: usize,
    categories: usize,
    specific_map: Vec<i32>,
    n_specific: usize,
    estimate_specific_means: bool,
    estimate_specific_vars: bool,
    packed_bits: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BifactorDecoded {
    pub general_slope: Vec<f64>,
    pub specific_slope: Vec<f64>,
    pub intercepts: Vec<f64>,
    pub general_means: Vec<f64>,
    pub general_sds: Vec<f64>,
    pub specific_means: Vec<f64>,
    pub specific_sds: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
impl BifactorPackedArchive {
    pub fn new(
        anchor: Vec<bool>,
        groups: usize,
        categories: usize,
        specific_map: Vec<i32>,
        n_specific: usize,
        estimate_specific_means: bool,
        estimate_specific_vars: bool,
        packed: Vec<f64>,
    ) -> Result<Self, String> {
        let a = Self {
            schema_version: 1,
            anchor,
            groups,
            categories,
            specific_map,
            n_specific,
            estimate_specific_means,
            estimate_specific_vars,
            packed_bits: packed.into_iter().map(f64::to_bits).collect(),
        };
        decode_bifactor(&a)?;
        Ok(a)
    }
    pub fn with_packed(&self, packed: Vec<f64>) -> Result<Self, String> {
        Self::new(
            self.anchor.clone(),
            self.groups,
            self.categories,
            self.specific_map.clone(),
            self.n_specific,
            self.estimate_specific_means,
            self.estimate_specific_vars,
            packed,
        )
    }
    pub fn anchor(&self) -> &[bool] {
        &self.anchor
    }
    pub fn specific_map(&self) -> &[i32] {
        &self.specific_map
    }
    pub fn n_specific(&self) -> usize {
        self.n_specific
    }
    pub fn estimate_specific_means(&self) -> bool {
        self.estimate_specific_means
    }
    pub fn estimate_specific_vars(&self) -> bool {
        self.estimate_specific_vars
    }
    pub fn packed_bits(&self) -> &[u64] {
        &self.packed_bits
    }
    pub fn packed_values(&self) -> Vec<f64> {
        self.packed_bits
            .iter()
            .copied()
            .map(f64::from_bits)
            .collect()
    }
    pub fn n_groups(&self) -> usize {
        self.groups
    }
    pub fn n_items(&self) -> usize {
        self.anchor.len()
    }
    pub fn n_categories(&self) -> usize {
        self.categories
    }
    pub fn coordinates(&self) -> Result<Vec<crate::mask_grm::Coordinate>, String> {
        if self.schema_version != 1 {
            return Err("invalid archive geometry/schema".into());
        }
        let coords = crate::mask_grm::bifactor_coordinates(
            &self.anchor,
            self.groups,
            self.categories,
            &self.specific_map,
            self.n_specific,
            self.estimate_specific_means,
            self.estimate_specific_vars,
        )?;
        if coords.len() != self.packed_bits.len() {
            return Err("coordinate length does not match anchor geometry".into());
        }
        Ok(coords)
    }
    pub fn to_json(&self) -> Result<String, String> {
        decode_bifactor(self)?;
        serde_json::to_string(self).map_err(|e| e.to_string())
    }
    pub fn from_json(s: &str) -> Result<Self, String> {
        let a: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        decode_bifactor(&a)?;
        Ok(a)
    }
}

pub fn decode_bifactor(a: &BifactorPackedArchive) -> Result<BifactorDecoded, String> {
    use crate::mask_grm::Coordinate as BifactorCoordinate;
    let coords = a.coordinates()?;
    let j = a.anchor.len();
    let z = a.categories - 1;
    let size = a.groups.checked_mul(j).ok_or("dense size overflow")?;
    let cells = size.checked_mul(z).ok_or("dense size overflow")?;
    let spec_cells = a
        .groups
        .checked_mul(a.n_specific)
        .ok_or("dense size overflow")?;
    if cells > 16_000_000 || spec_cells > 16_000_000 {
        return Err("dense workspace exceeds 16 million elements".into());
    }
    let mut d = BifactorDecoded {
        general_slope: vec![0.0; size],
        specific_slope: vec![0.0; size],
        intercepts: vec![0.0; cells],
        general_means: vec![0.0; a.groups],
        general_sds: vec![1.0; a.groups],
        specific_means: vec![0.0; spec_cells],
        specific_sds: vec![1.0; spec_cells],
    };
    for (c, &bits) in coords.iter().zip(&a.packed_bits) {
        let v = f64::from_bits(bits);
        if !v.is_finite() {
            return Err("nonfinite coordinate".into());
        }
        match *c {
            BifactorCoordinate::Slope(g, i) => d.general_slope[g * j + i] = v,
            BifactorCoordinate::SpecificSlope(g, i) => d.specific_slope[g * j + i] = v,
            BifactorCoordinate::FirstIntercept(g, i) => d.intercepts[(g * j + i) * z] = v,
            BifactorCoordinate::LogGap(g, i, k) => {
                let n = (g * j + i) * z + k;
                d.intercepts[n] = d.intercepts[n - 1] - v.exp();
            }
            BifactorCoordinate::Mean(g) => d.general_means[g] = v,
            BifactorCoordinate::LogSd(g) => d.general_sds[g] = v.exp(),
            BifactorCoordinate::SpecificMean(g, s) => {
                d.specific_means[g * a.n_specific + s] = v;
            }
            BifactorCoordinate::SpecificLogSd(g, s) => {
                d.specific_sds[g * a.n_specific + s] = v.exp();
            }
        }
    }
    for g in 1..a.groups {
        for (i, &common) in a.anchor.iter().enumerate() {
            if common {
                d.general_slope[g * j + i] = d.general_slope[i];
                d.specific_slope[g * j + i] = d.specific_slope[i];
                for k in 0..z {
                    d.intercepts[(g * j + i) * z + k] = d.intercepts[i * z + k];
                }
            }
        }
    }
    if d.general_sds
        .iter()
        .chain(&d.specific_sds)
        .any(|v| !v.is_finite() || *v <= 0.0)
        || d.general_means
            .iter()
            .chain(&d.specific_means)
            .chain(&d.general_slope)
            .chain(&d.specific_slope)
            .chain(&d.intercepts)
            .any(|v| !v.is_finite())
        || d.intercepts
            .chunks_exact(z)
            .any(|v| v.windows(2).any(|w| w[0] <= w[1]))
    {
        return Err("coordinate transform overflow/underflow".into());
    }
    Ok(d)
}
/// Owned copies of the native quadrature rule; count must be explicit.
pub fn quadrature(points: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (n, w) = crate::quadrature::require_gh_rule_unidim(points, "samefit quadrature")?;
    Ok((n.to_vec(), w.to_vec()))
}

#[derive(Deserialize)]
struct SavedInput {
    y: Vec<usize>,
    observed: Option<Vec<bool>>,
    group_id: Vec<usize>,
    n_groups: usize,
    n_cat: usize,
    item_ids: Vec<String>,
    q_general: usize,
}
/// Imported coordinates and request whose terminal likelihood agrees with the summary.
/// Historical identity is retained as provenance, not regenerated with a newer config type.
pub struct ImportedFit {
    archive: PackedArchive,
    input: SavedInput,
    loglik: f64,
    pub historical_identity: String,
}
impl ImportedFit {
    pub fn archive(&self) -> &PackedArchive {
        &self.archive
    }
    pub fn loglik(&self) -> f64 {
        self.loglik
    }
    pub fn responses(&self) -> &[usize] {
        &self.input.y
    }
    pub fn groups(&self) -> &[usize] {
        &self.input.group_id
    }
    pub fn item_ids(&self) -> &[String] {
        &self.input.item_ids
    }
    pub fn quadpts(&self) -> usize {
        self.input.q_general
    }
}
/// Hash-bound request plus explicit summary import; no optimization is called.
/// The summary must come from the caller's separately hash-verified result file.
pub fn import_summary(
    request: &str,
    request_sha256: &str,
    summary: &serde_json::Value,
    ll_atol: f64,
) -> Result<ImportedFit, String> {
    use sha2::{Digest, Sha256};
    if !ll_atol.is_finite() || ll_atol < 0.0 {
        return Err("invalid LL tolerance".into());
    }
    if format!("{:x}", Sha256::digest(request.as_bytes())) != request_sha256 {
        return Err("request SHA256 mismatch".into());
    }
    let input: SavedInput = serde_json::from_str(request).map_err(|e| e.to_string())?;
    let j = input.item_ids.len();
    let n = input.group_id.len();
    if j == 0
        || n == 0
        || input.y.len() != j.checked_mul(n).ok_or("input overflow")?
        || input
            .observed
            .as_ref()
            .is_some_and(|v| v.len() != input.y.len() || v.iter().any(|b| !*b))
        || input.group_id.iter().any(|&g| g >= input.n_groups)
        || input.y.iter().any(|&v| v >= input.n_cat)
        || input
            .item_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != j
    {
        return Err("invalid or incomplete saved input".into());
    }
    let anchor = serde_json::from_value(summary["anchor"].clone()).map_err(|e| e.to_string())?;
    let packed = serde_json::from_value(summary["packed"].clone()).map_err(|e| e.to_string())?;
    let archive = PackedArchive::new(anchor, input.n_groups, input.n_cat, packed)?;
    if archive.n_items() != j
        || serde_json::to_value(archive.coordinates()?).map_err(|e| e.to_string())?
            != summary["coordinate_map"]
        || summary["q_general"].as_u64() != Some(input.q_general as u64)
    {
        return Err("summary coordinate/order/quadrature mismatch".into());
    }
    let d = decode(&archive)?;
    let (nodes, weights) = quadrature(input.q_general)?;
    let z = input.n_cat - 1;
    let mut ll = 0.0;
    for (person, &g) in input.group_id.iter().enumerate() {
        let mut terms = Vec::with_capacity(nodes.len());
        for (&t, &w) in nodes.iter().zip(&weights) {
            let theta = d.means[g] + d.sds[g] * t;
            let mut v = w.ln();
            for i in 0..j {
                let lp = crate::poly::grm_logprobs(
                    d.slope[g * j + i] * theta,
                    &d.intercepts[(g * j + i) * z..(g * j + i + 1) * z],
                );
                v += lp[input.y[person * j + i]];
            }
            terms.push(v);
        }
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        ll += max + terms.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    }
    let expected = summary["loglik_cpu_f64"]
        .as_f64()
        .ok_or("missing terminal LL")?;
    if !ll.is_finite() || !expected.is_finite() || (ll - expected).abs() > ll_atol {
        return Err(format!("terminal LL mismatch: delta={}", ll - expected));
    }
    let identity = summary["fit_identity"]
        .as_str()
        .ok_or("missing historical identity")?
        .to_string();
    Ok(ImportedFit {
        archive,
        input,
        loglik: ll,
        historical_identity: identity,
    })
}

pub struct BifactorImportedFit {
    archive: BifactorPackedArchive,
    input: SavedInput,
    q_specific: usize,
    loglik: f64,
    pub historical_identity: String,
}
impl BifactorImportedFit {
    pub fn archive(&self) -> &BifactorPackedArchive {
        &self.archive
    }
    pub fn loglik(&self) -> f64 {
        self.loglik
    }
    pub fn responses(&self) -> &[usize] {
        &self.input.y
    }
    pub fn groups(&self) -> &[usize] {
        &self.input.group_id
    }
    pub fn item_ids(&self) -> &[String] {
        &self.input.item_ids
    }
    pub fn quadpts(&self) -> usize {
        self.input.q_general
    }
    pub fn specific_quadpts(&self) -> usize {
        self.q_specific
    }
}

pub fn import_bifactor_summary(
    request: &str,
    request_sha256: &str,
    summary: &serde_json::Value,
    ll_atol: f64,
) -> Result<BifactorImportedFit, String> {
    use sha2::{Digest, Sha256};
    if !ll_atol.is_finite() || ll_atol < 0.0 {
        return Err("invalid LL tolerance".into());
    }
    if format!("{:x}", Sha256::digest(request.as_bytes())) != request_sha256 {
        return Err("request SHA256 mismatch".into());
    }
    let input: SavedInput = serde_json::from_str(request).map_err(|e| e.to_string())?;
    let j = input.item_ids.len();
    let n = input.group_id.len();
    if j == 0
        || n == 0
        || input.y.len() != j.checked_mul(n).ok_or("input overflow")?
        || input
            .observed
            .as_ref()
            .is_some_and(|v| v.len() != input.y.len() || v.iter().any(|b| !*b))
        || input.group_id.iter().any(|&g| g >= input.n_groups)
        || input.y.iter().any(|&v| v >= input.n_cat)
        || input
            .item_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != j
    {
        return Err("invalid or incomplete saved input".into());
    }
    let anchor = serde_json::from_value(summary["anchor"].clone()).map_err(|e| e.to_string())?;
    let specific_map: Vec<i32> =
        serde_json::from_value(summary["specific_map"].clone()).map_err(|e| e.to_string())?;
    let n_specific = summary["n_specific"].as_u64().ok_or("missing n_specific")? as usize;
    let q_specific = summary["q_specific"].as_u64().ok_or("missing q_specific")? as usize;
    let estimate_specific_means = summary["estimate_specific_means"]
        .as_bool()
        .ok_or("missing estimate_specific_means")?;
    let estimate_specific_vars = summary["estimate_specific_vars"]
        .as_bool()
        .ok_or("missing estimate_specific_vars")?;
    let packed = serde_json::from_value(summary["packed"].clone()).map_err(|e| e.to_string())?;
    let archive = BifactorPackedArchive::new(
        anchor,
        input.n_groups,
        input.n_cat,
        specific_map,
        n_specific,
        estimate_specific_means,
        estimate_specific_vars,
        packed,
    )?;
    if archive.n_items() != j
        || serde_json::to_value(archive.coordinates()?).map_err(|e| e.to_string())?
            != summary["coordinate_map"]
        || summary["q_general"].as_u64() != Some(input.q_general as u64)
    {
        return Err("summary coordinate/order/quadrature mismatch".into());
    }
    let d = decode_bifactor(&archive)?;
    let (xg, wg) = quadrature(input.q_general)?;
    let (xs, ws) = if n_specific > 0 {
        quadrature(q_specific)?
    } else {
        (vec![0.0], vec![1.0])
    };
    let z = input.n_cat - 1;
    let mut ll = 0.0;
    for (person, &g) in input.group_id.iter().enumerate() {
        let mut terms_g = Vec::with_capacity(xg.len());
        for (&ng, &w_g) in xg.iter().zip(&wg) {
            let tg = d.general_means[g] + d.general_sds[g] * ng;
            let mut vg = w_g.ln();
            for i in 0..j {
                if archive.specific_map()[i] < 0 {
                    let lp = crate::poly::grm_logprobs(
                        d.general_slope[g * j + i] * tg,
                        &d.intercepts[(g * j + i) * z..(g * j + i + 1) * z],
                    );
                    vg += lp[input.y[person * j + i]];
                }
            }
            for s in 0..n_specific {
                let mut has_item = false;
                let mut terms_s = Vec::with_capacity(xs.len());
                for (&ns, &w_s) in xs.iter().zip(&ws) {
                    let ts = d.specific_means[g * n_specific + s]
                        + d.specific_sds[g * n_specific + s] * ns;
                    let mut vs = w_s.ln();
                    for i in 0..j {
                        if archive.specific_map()[i] == s as i32 {
                            has_item = true;
                            let lp = crate::poly::grm_logprobs(
                                d.general_slope[g * j + i] * tg + d.specific_slope[g * j + i] * ts,
                                &d.intercepts[(g * j + i) * z..(g * j + i + 1) * z],
                            );
                            vs += lp[input.y[person * j + i]];
                        }
                    }
                    terms_s.push(vs);
                }
                if has_item {
                    let max_s = terms_s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    vg += max_s + terms_s.iter().map(|v| (v - max_s).exp()).sum::<f64>().ln();
                }
            }
            terms_g.push(vg);
        }
        let max_g = terms_g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        ll += max_g + terms_g.iter().map(|v| (v - max_g).exp()).sum::<f64>().ln();
    }
    let expected = summary["loglik_cpu_f64"]
        .as_f64()
        .ok_or("missing terminal LL")?;
    if !ll.is_finite() || !expected.is_finite() || (ll - expected).abs() > ll_atol {
        return Err(format!("terminal LL mismatch: delta={}", ll - expected));
    }
    let identity = summary["fit_identity"]
        .as_str()
        .ok_or("missing historical identity")?
        .to_string();
    Ok(BifactorImportedFit {
        archive,
        input,
        q_specific,
        loglik: ll,
        historical_identity: identity,
    })
}
