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
        .count() as f64 / n;
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
    gates.push(Gate { name: "qwk", value: qwk, threshold: 0.70, pass: qwk >= 0.70 });
    let r = pearson_r(&auto_f, &human_f)?;
    gates.push(Gate { name: "pearson_r", value: r, threshold: 0.70, pass: r >= 0.70 });
    let s = smd(&auto_f, &human_f)?;
    gates.push(Gate { name: "smd", value: s, threshold: 0.15, pass: s.abs() <= 0.15 });

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
            let idx: Vec<usize> =
                (0..groups.len()).filter(|&i| groups[i] as usize == g).collect();
            if idx.len() < 2 {
                continue;
            }
            let ga: Vec<f64> = idx.iter().map(|&i| auto_f[i]).collect();
            let gh: Vec<f64> = idx.iter().map(|&i| human_f[i]).collect();
            if let Ok(gs) = smd(&ga, &gh) {
                if gs.abs() > worst.abs() {
                    worst = gs;
                }
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
    Ok(ValidationVerdict { gates, exact_agreement: exact, adjacent_agreement: adjacent, pass })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kappa_hand_computed_2x2() {
        // table: a\b -> [[20, 5], [10, 65]], n = 100
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (x, y, count) in [(0, 0, 20), (0, 1, 5), (1, 0, 10), (1, 1, 65)] {
            for _ in 0..count {
                a.push(x);
                b.push(y);
            }
        }
        // po = .85; pe = .25*.30 + .75*.70 = .60; kappa = .25/.40 = .625
        let k = cohen_kappa(&a, &b, 2).unwrap();
        assert!((k - 0.625).abs() < 1e-9, "kappa {k}");
        // binary QWK equals unweighted kappa
        let qwk = quadratic_weighted_kappa(&a, &b, 2).unwrap();
        assert!((qwk - k).abs() < 1e-9);
        let (exact, adjacent) = agreement_rates(&a, &b).unwrap();
        assert!((exact - 0.85).abs() < 1e-9);
        assert!((adjacent - 1.0).abs() < 1e-9, "binary adjacent is degenerate at 1");
    }

    #[test]
    fn smd_and_r_hand_computed() {
        let human = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let auto = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        // p_h = .625, sd_h = sqrt(.625*.375); p_a = .75
        let expect = (0.75 - 0.625) / (0.625_f64 * 0.375).sqrt();
        assert!((smd(&auto, &human).unwrap() - expect).abs() < 1e-9);
        let r = pearson_r(&auto, &human).unwrap();
        assert!(r > 0.6 && r < 1.0);
    }

    #[test]
    fn verdict_gates_flag_degradation() {
        // auto-human agreement clearly worse than human-human
        let human: Vec<u32> = (0..200).map(|i| (i % 2) as u32).collect();
        let auto: Vec<u32> =
            (0..200).map(|i| if i % 5 == 0 { 1 - (i % 2) as u32 } else { (i % 2) as u32 }).collect();
        let h2: Vec<u32> = human.clone(); // perfect human-human baseline
        let verdict =
            validate_scoring(&auto, &human, 2, Some((&human, &h2)), None).unwrap();
        let degr = verdict.gates.iter().find(|g| g.name == "degradation").unwrap();
        assert!(!degr.pass, "20% flips vs perfect baseline must flag degradation");
        assert!(verdict.exact_agreement < 1.0);
    }

    #[test]
    fn subgroup_smd_catches_biased_slice() {
        // group 1 systematically over-scored by the auto rater
        let mut auto = Vec::new();
        let mut human = Vec::new();
        let mut grp = Vec::new();
        let mut state = 9u64;
        let mut unif = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for i in 0..400 {
            let g = (i % 2) as u32;
            let h = if unif() < 0.5 { 1u32 } else { 0 };
            let a = if g == 1 && h == 0 && unif() < 0.5 { 1 } else { h };
            auto.push(a);
            human.push(h);
            grp.push(g);
        }
        let verdict = validate_scoring(&auto, &human, 2, None, Some(&grp)).unwrap();
        let sg = verdict.gates.iter().find(|g| g.name == "subgroup_smd").unwrap();
        assert!(!sg.pass, "inflated group-1 scores must flag the subgroup SMD gate");
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(cohen_kappa(&[0, 1], &[0], 2).is_err());
        assert!(quadratic_weighted_kappa(&[0, 0], &[0, 0], 2).is_err());
        assert!(pearson_r(&[1.0, 1.0], &[0.0, 1.0]).is_err());
        assert!(smd(&[1.0, 1.0], &[1.0, 1.0]).is_err());
        assert!(quadratic_weighted_kappa(&[0, 3], &[0, 1], 2).is_err());
    }
}
