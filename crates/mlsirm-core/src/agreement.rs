//! Machine-scoring validation statistics and acceptance gates.
//!
//! Implements the operational criteria of Williamson, Xi & Breyer (2012), "A
//! Framework for Evaluation and Use of Automated Scoring" (Educational
//! Measurement: Issues and Practice 31(1), 2-13) for validating an automated
//! scorer (here: an LLM-as-a-Judge) against human ratings:
//!
//! - quadratic-weighted kappa `κ_w = 1 - Σ w_ij O_ij / Σ w_ij E_ij` with
//!   `w_ij = (i-j)^2/(K-1)^2` (Fleiss & Cohen 1973); gate `κ_w >= .70`
//!   (collapses to Cohen's unweighted kappa for binary labels);
//! - Pearson r on the paired scores; gate `r >= .70`;
//! - degradation vs a human-human baseline `Δ = stat_hh - stat_ah`; gate
//!   `Δ <= .10`;
//! - standardized mean difference `SMD = (M_auto - M_human)/SD_human`; gate
//!   `|SMD| <= .15` overall and `<= .10` within every subgroup;
//! - exact (and adjacent) agreement: reported, explicitly NOT a gate.

/// Cross-tabulate two label vectors with values in `0..k`.
fn joint_counts(a: &[u32], b: &[u32], k: usize) -> Result<Vec<f64>, String> {
    if a.len() != b.len() || a.is_empty() {
        return Err("paired label vectors must be non-empty and equal-length".into());
    }
    let mut table = vec![0.0_f64; k * k];
    for (&x, &y) in a.iter().zip(b) {
        if x as usize >= k || y as usize >= k {
            return Err(format!("labels must be in 0..{k}"));
        }
        table[x as usize * k + y as usize] += 1.0;
    }
    Ok(table)
}

/// Weighted kappa with weights `w_ij = (i-j)^2/(K-1)^2` (quadratic; K >= 2).
/// For `k = 2` this equals Cohen's unweighted kappa.
pub fn quadratic_weighted_kappa(a: &[u32], b: &[u32], k: usize) -> Result<f64, String> {
    if k < 2 {
        return Err("kappa needs at least 2 categories".into());
    }
    let table = joint_counts(a, b, k)?;
    let n = a.len() as f64;
    let mut row = vec![0.0_f64; k];
    let mut col = vec![0.0_f64; k];
    for i in 0..k {
        for j in 0..k {
            row[i] += table[i * k + j];
            col[j] += table[i * k + j];
        }
    }
    let denom_w = ((k - 1) * (k - 1)) as f64;
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for i in 0..k {
        for j in 0..k {
            let w = ((i as f64 - j as f64) * (i as f64 - j as f64)) / denom_w;
            num += w * table[i * k + j] / n;
            den += w * (row[i] / n) * (col[j] / n);
        }
    }
    if den <= 0.0 {
        return Err("degenerate marginals: expected weighted disagreement is zero".into());
    }
    Ok(1.0 - num / den)
}

/// Cohen's unweighted kappa.
pub fn cohen_kappa(a: &[u32], b: &[u32], k: usize) -> Result<f64, String> {
    let table = joint_counts(a, b, k)?;
    let n = a.len() as f64;
    let mut po = 0.0_f64;
    let mut row = vec![0.0_f64; k];
    let mut col = vec![0.0_f64; k];
    for i in 0..k {
        po += table[i * k + i] / n;
        for j in 0..k {
            row[i] += table[i * k + j];
            col[j] += table[i * k + j];
        }
    }
    let pe: f64 = (0..k).map(|i| (row[i] / n) * (col[i] / n)).sum();
    if (1.0 - pe).abs() < 1e-12 {
        return Err("degenerate marginals: chance agreement is 1".into());
    }
    Ok((po - pe) / (1.0 - pe))
}

/// Pearson product-moment correlation of paired scores.
pub fn pearson_r(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() || a.len() < 2 {
        return Err("paired score vectors must be equal-length with n >= 2".into());
    }
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return Err("zero variance in one of the score vectors".into());
    }
    Ok(sab / (saa.sqrt() * sbb.sqrt()))
}

/// Standardized mean difference, standardized on the HUMAN score SD:
/// `(M_auto - M_human) / SD_human` (Williamson et al. criterion E).
pub fn smd(auto: &[f64], human: &[f64]) -> Result<f64, String> {
    if auto.len() != human.len() || human.len() < 2 {
        return Err("paired score vectors must be equal-length with n >= 2".into());
    }
    let n = human.len() as f64;
    let mh = human.iter().sum::<f64>() / n;
    let ma = auto.iter().sum::<f64>() / n;
    let var_h = human.iter().map(|&v| (v - mh) * (v - mh)).sum::<f64>() / n;
    if var_h <= 0.0 {
        return Err("human scores have zero variance".into());
    }
    Ok((ma - mh) / var_h.sqrt())
}

/// Proportion of exact matches, and matches within +/- 1 category.
pub fn agreement_rates(a: &[u32], b: &[u32]) -> Result<(f64, f64), String> {
    if a.len() != b.len() || a.is_empty() {
        return Err("paired label vectors must be non-empty and equal-length".into());
    }
    let n = a.len() as f64;
    let exact = a.iter().zip(b).filter(|(&x, &y)| x == y).count() as f64 / n;
    let adjacent = a
        .iter()
        .zip(b)
        .filter(|(&x, &y)| (x as i64 - y as i64).abs() <= 1)
        .count() as f64
        / n;
    Ok((exact, adjacent))
}

/// One gate outcome: the statistic, its threshold, and whether it passed.
#[derive(Clone, Debug)]
pub struct Gate {
    pub name: &'static str,
    pub value: f64,
    pub threshold: f64,
    pub pass: bool,
}

/// Conjunctive validation verdict per Williamson et al. (2012).
#[derive(Clone, Debug)]
pub struct ValidationVerdict {
    pub gates: Vec<Gate>,
    /// Reported-only statistics (exact/adjacent agreement).
    pub exact_agreement: f64,
    pub adjacent_agreement: f64,
    pub pass: bool,
}

/// Run the conjunctive acceptance gates on paired (auto, human) labels in
/// `0..k`. `human_human` optionally supplies a double-scored baseline
/// (pairs of human labels) for the degradation criterion; `subgroup` labels
/// each observation for the fairness SMD.
pub fn validate_scoring(
    auto: &[u32],
    human: &[u32],
    k: usize,
    human_human: Option<(&[u32], &[u32])>,
    subgroup: Option<&[u32]>,
) -> Result<ValidationVerdict, String> {
    let auto_f: Vec<f64> = auto.iter().map(|&v| v as f64).collect();
    let human_f: Vec<f64> = human.iter().map(|&v| v as f64).collect();
    let mut gates = Vec::new();

    let qwk = quadratic_weighted_kappa(auto, human, k)?;
    gates.push(Gate {
        name: "qwk",
        value: qwk,
        threshold: 0.70,
        pass: qwk >= 0.70,
    });
    let r = pearson_r(&auto_f, &human_f)?;
    gates.push(Gate {
        name: "pearson_r",
        value: r,
        threshold: 0.70,
        pass: r >= 0.70,
    });
    let s = smd(&auto_f, &human_f)?;
    gates.push(Gate {
        name: "smd",
        value: s,
        threshold: 0.15,
        pass: s.abs() <= 0.15,
    });

    if let Some((h1, h2)) = human_human {
        let hh = quadratic_weighted_kappa(h1, h2, k)?;
        let degradation = hh - qwk;
        gates.push(Gate {
            name: "degradation",
            value: degradation,
            threshold: 0.10,
            pass: degradation <= 0.10,
        });
    }

    if let Some(groups) = subgroup {
        if groups.len() != auto.len() {
            return Err("subgroup labels must match the paired vectors".into());
        }
        let n_groups = groups.iter().map(|&g| g as usize).max().unwrap_or(0) + 1;
        let mut worst: f64 = 0.0;
        for g in 0..n_groups {
            let idx: Vec<usize> = (0..groups.len())
                .filter(|&i| groups[i] as usize == g)
                .collect();
            if idx.len() < 2 {
                continue;
            }
            let ga: Vec<f64> = idx.iter().map(|&i| auto_f[i]).collect();
            let gh: Vec<f64> = idx.iter().map(|&i| human_f[i]).collect();
            let Ok(gs) = smd(&ga, &gh) else {
                continue;
            };
            if gs.abs() > worst.abs() {
                worst = gs;
            }
        }
        gates.push(Gate {
            name: "subgroup_smd",
            value: worst,
            threshold: 0.10,
            pass: worst.abs() <= 0.10,
        });
    }

    let (exact, adjacent) = agreement_rates(auto, human)?;
    let pass = gates.iter().all(|g| g.pass);
    Ok(ValidationVerdict {
        gates,
        exact_agreement: exact,
        adjacent_agreement: adjacent,
        pass,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/agreement_tests.rs"]
mod tests;
