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
